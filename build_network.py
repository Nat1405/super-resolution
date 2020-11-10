from sr_utils import *
from models import *


input_depth = 32
	 
INPUT =     'noise'
pad   =     'reflection'

tv_weight = 0.0

reg_noise_std = 0.03

def build_network(dtype):
	"""
	Imsize: (w,h) tuple
	"""

	NET_TYPE = 'skip' # UNet, ResNet
	net = get_net(input_depth, 'skip', pad,
	              skip_n33d=128, 
	              skip_n33u=128, 
	              skip_n11=4, 
	              num_scales=5,
	              upsample_mode='bilinear').type(dtype)

	return net
