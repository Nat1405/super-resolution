import common_utils as common
import state
import astropy.visualization
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import sys
from skimage.measure import compare_psnr
from skimage.measure import compare_mse

from PIL import Image
import PIL

import matplotlib.pyplot as plt
import argparse
import configparser
import astropy.io.fits as fits

import state


def get_background(noise_x, noise_y, region_size=200):
    """Gets an estimate of the stddevation of the image."""
    # For now, just do crude estimate from first image.
    return np.std(np.array(state.imgs[0]['orig_pil']))    

def compare_HR(ground_truth_HR, test_HR):
    # Calculates scalar loss between two np HR images.
    target_loss = compare_mse(ground_truth_HR, test_HR)
    return target_loss

def get_baselines(imgs):
    """Add psnr and target loss measures to imgs."""
    for img in imgs:
        img['psnr_bicubic'] = compare_psnr(np.array(img['HR_pil']), np.array(img['HR_bicubic']))
        img['psnr_bicubic_blurred'] = compare_psnr(np.array(img['HR_pil']), np.array(img['HR_bicubic_blurred']))
        img['target_loss_bicubic'] = compare_HR(np.array(img['HR_pil']), np.array(img['HR_bicubic']))
        img['target_loss_bicubic_blurred'] = compare_HR(np.array(img['HR_pil']), np.array(img['HR_bicubic_blurred']))

    print("PSNR: Bicubic {} / Blurred Bicubic {}".format(imgs[0]['psnr_bicubic'], imgs[0]['psnr_bicubic_blurred']))
    print("Target Loss: Bicubic {} / Blurred Bicubic {}".format(imgs[0]['target_loss_bicubic'], imgs[0]['target_loss_bicubic_blurred']))

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def blurImage(img):
    # Img: tensor of shape (X, Y)
    config = common.get_config()
    KERNEL_SIZE = config.getint('DEFAULT', 'blur_size')

    if config['DEFAULT']['blur'] == 'box':
        kernel = (1/(KERNEL_SIZE**2))*np.ones((1,1,KERNEL_SIZE,KERNEL_SIZE))
    elif config['DEFAULT']['blur'] == 'gauss':
        kernel = np.expand_dims(np.expand_dims(gkern(KERNEL_SIZE), axis=0), axis=0)
    elif config['DEFAULT']['blur'] == 'none':
        Interval = astropy.visualization.MinMaxInterval()
        return Image.fromarray(Interval(np.array(img)), mode='F')
    else:    
        print("Unrecognized blur")
        return

    m = nn.Conv2d(1, 1, KERNEL_SIZE, stride=1, padding=(int((KERNEL_SIZE-1)/2),int((KERNEL_SIZE-1)/2)), padding_mode='reflect')
    kernel = torch.Tensor(kernel)
    kernel = torch.nn.Parameter( kernel ) # calling this turns tensor into "weight" parameter
    m.weight = kernel

    with torch.no_grad():
        output = m(TF.to_tensor(img).unsqueeze(0))
    Interval = astropy.visualization.MinMaxInterval()
    return Image.fromarray(Interval(output.squeeze(0).numpy()[0]), mode='F')


def make_baseline_figure(HR, bicubic_HR, LR, name):
    # Make baseline comparision figure.
    # HR, Bicubic, LR (nearest-neighbor upsampled).
    config = common.get_config()
    m = nn.Upsample(scale_factor=config.getint("DEFAULT", "factor"), mode='nearest')
    up_LR = m(LR.unsqueeze(0)).squeeze(0)

    ncols = 3
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14,7))
    fig.suptitle('{}'.format(name), fontsize=16)

    axes[0].imshow(torch.flip(HR.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[0].set_title('HR')
    axes[0].set_xlabel('({}x{})'.format(HR.size()[1], HR.size()[2]))
    
    axes[1].imshow(torch.flip(bicubic_HR.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[1].set_title('HR_bicubic')
    axes[1].set_xlabel('({}x{})'.format(bicubic_HR.size()[1], bicubic_HR.size()[2]))
    
    axes[2].imshow(torch.flip(up_LR.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[2].set_title('LR (NN-Upsampled)')
    axes[2].set_xlabel('({}x{})'.format(LR.size()[1], LR.size()[2]))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(name)

def make_progress_figure(HR, HR_bicubic, HR_out, LR, LR_out, HR_name, LR_name):
    """Make two figures; one tracks HR progress, one tracks LR progress.
    HR, ..., LR_out: tensors of shape (1,H,W) or (1,H/factor,W/factor)"""
    
    # Make HR comparison figure
    ncols = 5
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14,7))
    fig.suptitle('{}\n({}x{})'.format(HR_name, HR.size()[1], HR.size()[2]), fontsize=16)

    axes[0].imshow(torch.flip(HR.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[0].set_title('HR')
    # Make HR output plot; include PSNR
    axes[1].imshow(torch.flip(HR_out.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[1].set_title('HR Output')
    psnr_HR = compare_psnr(HR.numpy(), HR_out.numpy())
    axes[1].set_xlabel('PSNR HR: {:.2f}'.format(psnr_HR))

    axes[2].imshow(torch.flip(HR_bicubic.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[2].set_title('HR_bicubic')
    psnr_bicubic = compare_psnr(HR.numpy(), HR_bicubic.numpy())
    axes[2].set_xlabel('PSNR Bicubic: {:.2f}'.format(psnr_bicubic))

    axes[3].imshow(torch.flip((HR_bicubic-HR).permute(1,2,0), dims=(0,)), cmap='gray')
    axes[3].set_title('Residual: \nBicubic - HR')
    bicubic_loss = compare_mse(HR.numpy(), HR_bicubic.numpy())
    axes[3].set_xlabel('MSE HR / Bicubic: {:.2e}'.format(bicubic_loss))

    axes[4].imshow(torch.flip((HR_out-HR).permute(1,2,0), dims=(0,)), cmap='gray')
    axes[4].set_title('Residual: \nHR Output - HR')
    target_loss = compare_mse(HR.numpy(), HR_out.numpy())
    axes[4].set_xlabel('MSE HR / HR Output: {:.2e}'.format(target_loss))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(HR_name)

    # Make LR comparison figure
    ncols = 3
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14,7))
    fig.suptitle('{}\n({}x{})'.format(LR_name, LR.size()[1], LR.size()[2]), fontsize=16)

    axes[0].imshow(torch.flip(LR.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[0].set_title('LR')
    
    axes[1].imshow(torch.flip(LR_out.permute(1,2,0), dims=(0,)), cmap='gray')
    axes[1].set_title('LR Output')
    psnr_LR = compare_psnr(LR.numpy(), LR_out.numpy())
    axes[1].set_xlabel('PSNR LR: {:.2f}'.format(psnr_LR))

    axes[2].imshow(torch.flip((LR_out-LR).permute(1,2,0), dims=(0,)), cmap='gray')
    axes[2].set_title('Residual: \nLR Output - LR')
    training_loss = compare_mse(LR.numpy(), LR_out.numpy())
    axes[2].set_xlabel('MSE LR / LR Output: {:.2e}'.format(training_loss))

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(LR_name)


def save_results():
    state.net.eval()
    # Save output data as fits files.

    for j in range(len(state.imgs)):
        hdu = fits.PrimaryHDU(np.array(state.imgs[j]['LR_pil']))
        hdu.writeto('output/LR_np_{}.fits'.format(j))
        common.saveFigure('output/LR_Ground_Truth_{}.png'.format(j), hdu.data)

        hdu = fits.PrimaryHDU(np.array(state.imgs[j]['HR_pil']))
        hdu.writeto('output/HR_np_{}.fits'.format(j))
        common.saveFigure('output/HR_Ground_Truth_{}.png'.format(j), hdu.data)

        hdu = fits.PrimaryHDU(np.array(state.imgs[j]['HR_bicubic']))
        hdu.writeto('output/HR_bicubic_{}.fits'.format(j))
        common.saveFigure('output/HR_bicubic_{}.png'.format(j), hdu.data)

        bicubic_residual = np.array(state.imgs[j]['HR_bicubic']) - np.array(state.imgs[j]['HR_pil'])
        common.saveFigure('output/HR_bicubic_residual_{}.png'.format(j), bicubic_residual)

        with torch.no_grad():
            state.net.eval()
            data = state.net(state.imgs[j]['net_input']).cpu()
        hdu = fits.PrimaryHDU(data)
        hdu.writeto('output/network_output_{}.fits'.format(j))
        common.saveFigure('output/HR_Output_{}.png'.format(j), hdu.data[0,0])
        output_residual = hdu.data[0,0] - np.array(state.imgs[j]['HR_pil'])
        common.saveFigure('output/Output_Residual_{}.png'.format(j), output_residual)

def printMetrics():
    print("Max PSNR HR: {}".format(max(state.history['psnr_HR'])))
    print("Max PSNR LR: {}".format(max(state.history['psnr_LR'])))


def makeInterpolation(imgs):
    if len(imgs) < 2:
        raise ValueError("Can't interpolate less than two images.")

    first_noise = imgs[0]['net_input']
    last_noise = imgs[-1]['net_input']

    n = len(imgs)

    delta = (last_noise - first_noise) / ( n - 1 )

    for j in range(len(imgs)):
        imgs[j]['net_input'] = first_noise + (j*delta)


def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out


def load_LR_HR_imgs_sr(fname):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: (width, height)
        crop_coordinates: (left, upper)
    '''
    config = common.get_config()

    # Load fits file to [0,1] normalized torch tensor with shape (1,H,W)
    orig_torch = common.get_image(fname)
    #orig_pil_blurred = blurImage(orig_pil)

    HR_torch = crop(orig_torch)
    #HR_pil_blurred = crop(orig_pil_blurred)

    # Create low resolution
    LR_torch = downsample(HR_torch)
    #LR_pil_blurred = downsample(HR_pil_blurred)

    print('HR and LR resolutions: %s, %s' % (str(HR_torch.size()), str(LR_torch.size())))

    input_depth = config.getint('DEFAULT', 'input_depth')
    imsize = config.getint('DEFAULT', 'imsize')
    net_input = common.get_noise(input_depth, 'noise', imsize).type(state.dtype)

    # Create bicubic upsampled versions of LR images for reference
    m = nn.Upsample(scale_factor=config.getint("DEFAULT", "factor"), mode='bicubic')
    HR_torch_bicubic = m(LR_torch.unsqueeze(0)).squeeze(0)
    #HR_bicubic_blurred = LR_pil_blurred.resize(HR_pil_blurred.size, Image.BICUBIC)

    out =   {
            'orig_torch': orig_torch,
            #'orig_pil_blurred': orig_pil_blurred,
            'HR_torch': HR_torch,
            #'HR_pil_blurred': HR_pil_blurred,
            'LR_torch': LR_torch,
            #'LR_pil_blurred': LR_pil_blurred,
            'net_input': net_input,
            'HR_torch_bicubic': HR_torch_bicubic,
            #'HR_bicubic_blurred': HR_bicubic_blurred
        }

    return out


def crop(orig_torch):
    """Crops a torch tensor in [0..1.], (1,H,W).
        crop_x: left x of cropping
        crop_y: lower y of cropping
    """
    config = common.get_config()
    crops = {
        "crop_x": config.getint('DEFAULT', 'crop_x'),
        "crop_y":  config.getint('DEFAULT', 'crop_y'),
    }

    imsize = config.getint('DEFAULT', 'imsize')
    # Crop the image
    HR_torch = orig_torch[:, 
        crops['crop_y']:crops['crop_y']+imsize,
        crops['crop_x']:crops['crop_x']+imsize
    ]
    return HR_torch

def downsample(HR_torch):
    """Downsample an image using average pooling.
        HR_torch: tensor image with shape (1,H,W), in [0..1].
    """
    config = common.get_config()
    factor = config.getint('DEFAULT', 'factor')

    m = nn.AvgPool2d(factor)
    LR_torch = m(HR_torch)
    return LR_torch

def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))