from __future__ import print_function

import argparse
import os, shutil
import glob
from PIL import Image
import numpy as np
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import astropy.visualization
import astropy.io.fits as fits
import configparser

import state
import common_utils as common
import sr_utils
from build_closure import build_closure
from build_network import build_network

# Use GPU if available
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    state.dtype = dtype
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor
    state.dtype = dtype 

# Setup space to save checkpoints and inputs
if os.path.exists('output'):
    shutil.rmtree("output")
os.mkdir("output")
os.mkdir("output/checkpoints")
os.mkdir("output/inputs")

# Setup tensorboard summary writer
writer = SummaryWriter(log_dir='output/runs/')

# Read config file
config = common.get_config()

# Load images. Crop to produce HR, downsample to produce LR,
# create bicubic reference frames.
state.imgs = []
for im_path in glob.glob(config['DEFAULT']['path_to_images']):
    state.imgs.append(sr_utils.load_LR_HR_imgs_sr(im_path))

# Get baselines, such as psnr and target loss of bicubic.
sr_utils.get_baselines(state.imgs)

# Make input vectors linear interpolation from first to last
if config.getboolean('DEFAULT', 'interpolate_input'):
    sr_utils.makeInterpolation(state.imgs)

# Finished modifying inputs; now save them.
for j in range(len(state.imgs)):
    torch.save(state.imgs[j]['net_input'], "output/inputs/input_{}.pt".format(j))

state.net = build_network(config.getint('DEFAULT', 'imsize'), dtype)

c = build_closure(writer, dtype)

state.i = 0
p = [x for x in state.net.parameters()]

print("Iteration / Frame used / psnr_LR / psnr_HR")
print(config['DEFAULT']['OPTIMIZER'])
common.optimize(config['DEFAULT']['OPTIMIZER'], p, c, config.getfloat('DEFAULT', 'LR'), config.getint('DEFAULT', 'num_iter'))


sr_utils.save_results()

sr_utils.printMetrics()

