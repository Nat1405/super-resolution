from __future__ import print_function

import argparse
import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import astropy.visualization
import configparser

import state
import common_utils
from sr_utils import *
from build_closure import build_closure
from build_network import build_network

# Use GPU if available
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor 


# Setup tensorboard summary writer
writer = SummaryWriter()

# Read config file
config = common_utils.get_config()

state.imgs = []
for im_path in glob.glob(config['DEFAULT']['path_to_images']):
    state.imgs.append(load_LR_HR_imgs_sr(im_path, 
    	(config.getint('DEFAULT', 'imsize'), config.getint('DEFAULT', 'imsize')), 
    	(config.getint('DEFAULT', 'crop_x'), config.getint('DEFAULT', 'crop_y')),
    	4, 
    	dtype))

state.net = build_network(config.getint('DEFAULT', 'imsize'), dtype)

c = build_closure(writer, dtype)

state.i = 0
p = [x for x in state.net.parameters()]

print("Iteration / Frame used / psnr_LR / psnr_HR")
print(config['DEFAULT']['OPTIMIZER'])
optimize(config['DEFAULT']['OPTIMIZER'], p, c, config.getfloat('DEFAULT', 'LR'), config.getint('DEFAULT', 'num_iter'))

out_HR = torch_to_np(state.net(state.imgs[0]['net_input']))[0]
plt.figure()
plt.imshow(out_HR)
plt.savefig("network_output.png")

