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
if config.has_section('LOADING') and config.getboolean('LOADING', 'load_precreated'):
    sr_utils.preload_LR_HR(
        config['LOADING']['path_to_LR'],
        config['LOADING']['path_to_HR']
    )
else: 
    for im_path in glob.glob(config['DEFAULT']['path_to_images']):
        state.imgs.append(sr_utils.load_LR_HR_imgs_sr(im_path))

# Get baselines, such as psnr and target loss of bicubic.
#sr_utils.get_baselines(state.imgs)

# Make input vectors linear interpolation from first to last
if config.getboolean('DEFAULT', 'interpolate_input'):
    sr_utils.makeInterpolation(state.imgs)

# Finished modifying inputs; now save them. Also make baseline figures.
for j in range(len(state.imgs)):
    torch.save(state.imgs[j]['net_input'], "output/inputs/input_{}.pt".format(j))
    sr_utils.make_baseline_figure(
        state.imgs[j]['HR_torch'].cpu(),
        state.imgs[j]['HR_torch_bicubic'].cpu(),
        state.imgs[j]['LR_torch'].cpu(),
        'baseline_{}'.format(j)
    )

state.net = build_network(state.dtype)

c = build_closure(writer, state.dtype)

state.i = 0
p = [x for x in state.net.parameters()]

sr_utils.printMetrics()
print("Iteration / Frame used / psnr_LR / psnr_HR")
print(config['DEFAULT']['OPTIMIZER'])
common.optimize(config['DEFAULT']['OPTIMIZER'], p, c, config.getfloat('DEFAULT', 'LR'), config.getint('DEFAULT', 'num_iter'))


sr_utils.save_results()

