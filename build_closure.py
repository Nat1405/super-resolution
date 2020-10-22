from sr_utils import *
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
import astropy.visualization
import collections
from torchvision.models import vgg
import torchvision.transforms as transforms

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
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.cfg')

    plot_steps_low = config.getint('DEFAULT', 'plot_steps_low')
    plot_steps_high = config.getint('DEFAULT', 'plot_steps_high')

    blur = (config['DEFAULT']['blur'] != 'none')

    mse = torch.nn.MSELoss().type(dtype)
    downsampler = Downsampler(n_planes=1, factor=4, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    loss_network = None
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

    def get_loss(out_LR, ground_truth_LR):
        """Calculates loss from the low-resolution output of the network.
        """
        if loss_network:
            out_LR = out_LR.repeat(3, 1, 1)
            ground_truth_LR = ground_truth_LR.repeat(3,1,1)
            total_loss = mse(
                    loss_network(grey_to_normal_rgb(out_LR)).relu3_3,
                    loss_network(grey_to_normal_rgb(ground_truth_LR)).relu3_3
                )
        else:
            total_loss = mse(out_LR, ground_truth_LR)

        return total_loss

    def get_scalars(out_LR, out_HR, ground_truth_LR, ground_truth_HR):
        """Inputs are all torch tensors.
        Calculate:
            Target Loss: mse(out_HR, ground_truth_HR)
            PSNR_HR: psnr(out_HR, ground_truth_HR)
            PSNR_LR: psnr(out_LR, ground_truth_LR)
        """
        psnr_LR = compare_psnr(np.array(ground_truth_LR), common.torch_to_np(out_LR.unsqueeze(0)))
        psnr_HR = compare_psnr(np.array(ground_truth_HR), common.torch_to_np(out_HR.unsqueeze(0)))
        target_loss = compare_mse(np.array(ground_truth_LR), common.torch_to_np(out_LR.unsqueeze(0)))

        return target_loss, psnr_LR, psnr_HR

    def get_images(before_HR, after_HR, before_LR, after_LR):
        HR_grid = torchvision.utils.make_grid([
            torch.clamp(before_HR, 0, 1), torch.clamp(after_HR, 0, 1)
            ], 2)
        LR_grid = torchvision.utils.make_grid([
            torch.clamp(before_LR, 0, 1), torch.clamp(after_LR, 0, 1)
            ], 2)
        
        return HR_grid, LR_grid

    def closure():
        # Train with a different input/output pair at each iteration.
        index = state.i % len(state.imgs)
        net_input = state.imgs[index]['net_input']
        ground_truth_LR = TF.to_tensor(state.imgs[index]['LR_pil']).type(state.dtype)
        ground_truth_HR = TF.to_tensor(state.imgs[index]['HR_pil']).type(state.dtype)
        
        # Feed through actual network
        out_HR = state.net(net_input)
        out_LR = downsampler(out_HR)

        out_HR = out_HR.squeeze(0)
        out_LR = out_LR.squeeze(0)

        # Get loss and train
        total_loss = get_loss(out_LR, ground_truth_LR)
        total_loss.backward()

        if (state.i % plot_steps_low == 0) and (index == 0):
            target_loss, psnr_LR, psnr_HR = get_scalars(
                out_LR, 
                out_HR,
                ground_truth_LR,
                ground_truth_HR
                )
            
            print("{} {} {} {}".format(state.i, index, psnr_LR, psnr_HR))

            # TensorBoard History
            writer.add_scalar('PSNR LR', psnr_LR, state.i)
            writer.add_scalar('PSNR HR', psnr_HR, state.i)
            writer.add_scalar('Training Loss', total_loss.item(), state.i)
            writer.add_scalar('Target Loss', target_loss, state.i)

        if (state.i % plot_steps_high == 0) and (index == 0):
            # Lower frequency capturing of large data
            # Save parameters
            for name, param in state.net.named_parameters():
                writer.add_histogram('Parameter {}'.format(name), param.flatten(), state.i)
            
            # Add images
            HR_grid, LR_grid = get_images(
                ground_truth_HR.detach().clone(), 
                out_HR, 
                ground_truth_LR.detach().clone(), 
                out_LR)
            writer.add_image('Network LR Output', LR_grid, state.i)
            writer.add_image('Network HR Output', HR_grid, state.i)

            # Save a checkpoint
            torch.save(state.net, "./output/checkpoints/checkpoint_{}.pt".format(state.i))
        
        state.i += 1
        
        return total_loss

    return closure

