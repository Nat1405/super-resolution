import sr_utils
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
import astropy.visualization
import collections
import torchvision
from torchvision.models import vgg
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np

from models.downsampler import Downsampler
import state
import common_utils as common

LossOutput = collections.namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

def build_closure(writer, dtype):
    # Read config file
    config = common.configparser.ConfigParser()
    config.sections()
    config.read('config.cfg')

    plot_steps_low = config.getint('DEFAULT', 'plot_steps_low')
    plot_steps_high = config.getint('DEFAULT', 'plot_steps_high')

    blur = (config['DEFAULT']['blur'] != 'none')

    mse = torch.nn.MSELoss().type(dtype)
    downsampler = Downsampler(n_planes=config.getint('DEFAULT', 'n_channels'), factor=4, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    loss_network = None

    augmented_history = config.has_option('LOADING', 'augmented_history') and \
                        config.getboolean('LOADING', 'augmented_history')

    ignore_ground_truth = config.has_option('LOADING', 'ignore_ground_truth') and \
                          config.getboolean('LOADING', 'ignore_ground_truth')

    if config.has_option('DEFAULT', 'noise_level'):
        noise_level = float(config['DEFAULT']['noise_level'])
    else:
        noise_level = 0.03
    """
    if config.getboolean('DEFAULT', 'use_perceptual_loss'):
        vgg_model = vgg.vgg16(pretrained=True)
        if torch.cuda.is_available():
            vgg_model.cuda()
        loss_network = LossNetwork(vgg_model)
        loss_network.eval()

        # Pre-trained model expects normalized, RGB images. Here's a transform.
        grey_to_normal_rgb = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Lambda(lambda x: x.unsqueeze(0))
            ])

    noise = config['DEFAULT']['added_noise'] != 'none'
    if noise:
        noise_x = config.getint('DEFAULT', 'noise_x')
        noise_y = config.getint('DEFAULT', 'noise_y')
        background = sr_utils.get_background(noise_x, noise_y)"""

    def get_loss(out_LR, ground_truth_LR, blurred_LR):
        """Calculates loss from the low-resolution output of the network.
        """

        if blur:
            used_image = blurred_LR
        else:
            used_image = ground_truth_LR

        if loss_network:
            out_LR = out_LR.repeat(3, 1, 1)
            used_image = used_image.repeat(3,1,1)
            total_loss = mse(
                    loss_network(grey_to_normal_rgb(out_LR)).relu3_3,
                    loss_network(grey_to_normal_rgb(used_image)).relu3_3
                )
        else:
            total_loss = mse(out_LR, used_image)

        return total_loss

    def get_images(before_HR, after_HR, bicubic_HR, blurred_HR, before_LR, after_LR, blurred_LR):
        HR_grid = torchvision.utils.make_grid([
            torch.clamp(torch.flip(blurred_HR, [1,0]), 0, 1),
            torch.clamp(torch.flip(before_HR, [1, 0]), 0, 1), 
            torch.clamp(torch.flip(after_HR, [1, 0]), 0, 1),
            torch.clamp(torch.flip(bicubic_HR, [1, 0]), 0, 1),
            torch.clamp(torch.flip(common.makeResidual(after_HR, before_HR), [1, 0]), 0, 1)
            ], 5)
        LR_grid = torchvision.utils.make_grid([
            torch.clamp(torch.flip(blurred_LR, [1,0]), 0, 1),
            torch.clamp(torch.flip(before_LR, [1, 0]), 0, 1), 
            torch.clamp(torch.flip(after_LR, [1, 0]), 0, 1),
            torch.clamp(torch.flip(common.makeResidual(after_LR, before_LR), [1, 0]), 0, 1)
            ], 4)
        
        return HR_grid, LR_grid

    def closure():
        # Train with a different input/output pair at each iteration.
        index = state.i % len(state.imgs)
        net_input = state.imgs[index]['net_input']
        ground_truth_LR = state.imgs[index]['LR_torch']
        ground_truth_HR = state.imgs[index]['HR_torch']
        bicubic_HR = state.imgs[index]['HR_torch_bicubic']
        #blurred_HR = TF.to_tensor(state.imgs[index]['HR_pil_blurred']).type(state.dtype)
        #blurred_LR = TF.to_tensor(state.imgs[index]['LR_pil_blurred']).type(state.dtype)
        blurred_LR = ground_truth_LR
        blurred_HR = ground_truth_HR
        background = None
        noise = None

        # Feed through actual network
        net_input_noisy = net_input.detach().clone()
        noise_t = net_input.detach().clone()
        net_input_noisy = net_input_noisy + (noise_t.normal_() * noise_level)
        out_HR = state.net(net_input_noisy)
        if noise:
            with torch.no_grad():
                out_HR = out_HR + background*torch.randn(out_HR.size()).type(state.dtype)
        out_LR = downsampler(out_HR)

        out_HR = out_HR.squeeze(0)
        out_LR = out_LR.squeeze(0)

        # Get loss and train
        total_loss = get_loss(out_LR, ground_truth_LR, blurred_LR)
        total_loss.backward()

        out_HR = out_HR.detach().cpu()
        out_LR = out_LR.detach().cpu()
        ground_truth_LR = ground_truth_LR.cpu()
        ground_truth_HR = ground_truth_HR.cpu()
        bicubic_HR = bicubic_HR.cpu()
        blurred_LR = blurred_LR.cpu()
        blurred_HR = blurred_HR.cpu()

        if (state.i % plot_steps_low < len(state.imgs)):
            psnr_LR = compare_psnr(common.torch_to_np(ground_truth_LR), common.torch_to_np(out_LR))
            if not ignore_ground_truth:
                psnr_HR = compare_psnr(common.torch_to_np(ground_truth_HR), common.torch_to_np(out_HR))
                target_loss = sr_utils.compare_HR(common.torch_to_np(ground_truth_HR), common.torch_to_np(out_HR))
                state.imgs[index]['history_low'].psnr_HR.append(psnr_HR)
                state.imgs[index]['history_low'].target_loss.append(target_loss)
            else:
                psnr_HR = 0
                target_loss = 0
            state.imgs[index]['history_low'].iteration.append(state.i)
            state.imgs[index]['history_low'].psnr_LR.append(psnr_LR)
            state.imgs[index]['history_low'].training_loss.append(total_loss.item())

            if augmented_history:
                HR_torch_blurred = state.imgs[index]['HR_torch_blurred'].detach().cpu()
                LR_torch_downsampled = state.imgs[index]['LR_torch_downsampled'].detach().cpu()
                psnr_blurred = compare_psnr(common.torch_to_np(HR_torch_blurred), common.torch_to_np(out_HR))
                psnr_downsampled = compare_psnr(common.torch_to_np(LR_torch_downsampled), common.torch_to_np(out_LR))
                state.imgs[index]['history_low'].psnr_blurred.append(psnr_blurred)
                state.imgs[index]['history_low'].psnr_downsampled.append(psnr_downsampled)
                print("{} {} {} {} {} {}".format(state.i, index, psnr_LR, psnr_HR, psnr_blurred, psnr_downsampled))
            elif not ignore_ground_truth:
                print("{} {} {} {}".format(state.i, index, psnr_LR, psnr_HR))
            else:
                print("{} {} {} {}".format(state.i, index, psnr_LR, total_loss.item()))

            # TensorBoard History
            writer.add_scalar('PSNR LR', psnr_LR, state.i)
            writer.add_scalar('PSNR HR', psnr_HR, state.i)
            writer.add_scalar('Training Loss', total_loss.item(), state.i)
            writer.add_scalar('Target Loss', target_loss, state.i)

        if (state.i % plot_steps_high < len(state.imgs)):
            # Lower frequency capturing of large data
            psnr_LR = compare_psnr(common.torch_to_np(ground_truth_LR), common.torch_to_np(out_LR))
            if not ignore_ground_truth:
                psnr_HR = compare_psnr(common.torch_to_np(ground_truth_HR), common.torch_to_np(out_HR))
                target_loss = sr_utils.compare_HR(common.torch_to_np(ground_truth_HR), common.torch_to_np(out_HR))
                state.imgs[index]['history_high'].psnr_HR.append(psnr_HR)
                state.imgs[index]['history_high'].target_loss.append(target_loss)
            else:
                psnr_HR = 0
                target_loss = 0
            state.imgs[index]['history_high'].iteration.append(state.i)
            state.imgs[index]['history_high'].psnr_LR.append(psnr_LR)
            state.imgs[index]['history_high'].training_loss.append(total_loss.item())

            if augmented_history:
                HR_torch_blurred = state.imgs[index]['HR_torch_blurred'].detach().cpu()
                LR_torch_downsampled = state.imgs[index]['LR_torch_downsampled'].detach().cpu()
                psnr_blurred = compare_psnr(common.torch_to_np(HR_torch_blurred), common.torch_to_np(out_HR))
                psnr_downsampled = compare_psnr(common.torch_to_np(LR_torch_downsampled), common.torch_to_np(out_LR))
                state.imgs[index]['history_high'].psnr_blurred.append(psnr_blurred)
                state.imgs[index]['history_high'].psnr_downsampled.append(psnr_downsampled)

            # Save parameters
            for name, param in state.net.named_parameters():
                writer.add_histogram('Parameter {}'.format(name), param.flatten(), state.i)
            
            # Add images
            HR_grid, LR_grid = get_images(
                ground_truth_HR, 
                out_HR,
                bicubic_HR,
                blurred_HR,
                ground_truth_LR, 
                out_LR,
                blurred_LR
                )
            writer.add_image('Network LR Output', LR_grid, state.i)
            writer.add_image('Network HR Output', HR_grid, state.i)

            # Save a checkpoint
            #torch.save(state.net, "./output/checkpoints/checkpoint_{}.pt".format(state.i))
            #common.saveFigure("./output/checkpoints/checkpoint_{}.png".format(state.i), common.torch_to_np(out_HR))
            sr_utils.make_progress_figure(
                ground_truth_HR,
                bicubic_HR,
                out_HR,
                ground_truth_LR,
                out_LR,
                'HR_update_frame_{}_{}'.format(index, state.i),
                'LR_update_frame_{}_{}'.format(index, state.i)
            )
        state.i += 1
        
        return total_loss

    return closure

