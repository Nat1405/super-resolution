import torch
import torch.nn as nn
import torchvision
import sys, os

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import configparser
import astropy.io.fits as fits
import astropy.visualization

import state

def makeResidual(after_HR, before_HR):
    """Makes a residual image from two grayscale tensors by:
        - Converting two tensors to numpy
        - Subtracting them
        - Normalizing to [0..1]
        - Converting to tensors and returning them
    """
    after = torch_to_np(after_HR)
    before = torch_to_np(before_HR)
    result = after - before
    Interval = astropy.visualization.MinMaxInterval()
    if len(result.shape) == 2:
        result = np.expand_dims(result, axis=0)
    result = torch.tensor(Interval(result)).type(state.dtype).cpu()
    return result


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.sections()
    if args.config:
        config.read(args.config)
    else:
        config.read("config.cfg")
    return config

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def get_image(path):
    """Loads a fits file to a torch tensor normalized to [0,1], with shape (1,H,W)
    Loads a png file to a 3 channel tensor normalized to [0,1], with shape (3,H,W)
    """
    file_extension = os.path.split(path)[-1].split('.')[-1]
    if file_extension == "png":
        img = Image.open(path)
        img_np = np.flipud(np.array(img))
        if len(img_np.shape) == 3 and img_np.shape[-1] == 4:
            img_np = img_np[:,:,:3]
        Interval = astropy.visualization.MinMaxInterval()
        normed_img_np = Interval(img_np)
        t = torch.from_numpy(np.moveaxis(normed_img_np, -1, 0).copy()).type(state.dtype)
        return t
    elif file_extension == "fits":
        with fits.open(path) as hdul:
            Interval = astropy.visualization.MinMaxInterval()
            img_np = Interval((hdul['SCI'].data.astype(np.float32))).copy()
            img = torch.from_numpy(np.expand_dims(img_np, axis=0)).type(state.dtype)
        return img
    else:
        raise ValueError("{} is an invalid file type.".format(path))



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From C x W x H [0..1] to  either W x H [0..1] or 3 x W x H [0..1]
    '''
    if img_var.shape[0] == 1:
        return img_var.detach().cpu().numpy()[0]
    elif img_var.shape[0] == 3:
        return img_var.detach().cpu().numpy()
    else:
        raise ValueError("Invalid shape!")

def saveFigure(path, image):
    config = get_config()
    x1_low = config.getint("PLOTS", "x1_low")
    x1_high = config.getint("PLOTS", "x1_high")
    y1_low = config.getint("PLOTS", "y1_low")
    y1_high = config.getint("PLOTS", "y1_high")
    x2_low = config.getint("PLOTS", "x2_low")
    x2_high = config.getint("PLOTS", "x2_high")
    y2_low = config.getint("PLOTS", "y2_low")
    y2_high = config.getint("PLOTS", "y2_high")

    fig, ax = plt.subplots(figsize=[5, 7])
    ax.imshow(image, origin="lower", cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    axins = ax.inset_axes([0.5, -0.5, 0.5, 0.5])
    axins.imshow(image, origin="lower", cmap='gray')
    axins.set_xlim(x1_low,x1_high)
    axins.set_ylim(y1_low,y1_high)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor='r')
    
    axins2 = ax.inset_axes([0.0, -0.5, 0.5, 0.5])
    axins2.imshow(image, origin="lower", cmap='gray')
    axins2.set_xlim(x2_low,x2_high)
    axins2.set_ylim(y2_low,y2_high)
    axins2.set_xticks([])
    axins2.set_yticks([])
    ax.indicate_inset_zoom(axins2, edgecolor='b')

    fig.suptitle(path.split(os.path.sep)[-1])

    plt.savefig(path)

def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False