import torch
import torch.nn as nn
import numpy as np
import sr_utils
import astropy.io.fits as fits


CHANNELS = 1
WIDTH = 256
HEIGHT = 256
KERNEL_SIZE = 3

dtype = torch.FloatTensor

m = nn.Conv2d(1, 1, KERNEL_SIZE, stride=1, padding=(int((KERNEL_SIZE-1)/2),int((KERNEL_SIZE-1)/2)), padding_mode='reflect')
print(m.weight)

box_kernel = (1/(KERNEL_SIZE**2))*np.ones((1,1,KERNEL_SIZE,KERNEL_SIZE))


edge_detector = [[[
			[-1,-1,-1],
			[-1,8,-1],
			[-1,-1,-1]
]]]

import numpy as np
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

gaussian_kernel = np.expand_dims(np.expand_dims(gkern(KERNEL_SIZE), axis=0), axis=0)

# Reset m's weights to be the convolution filter we want
kernel = torch.Tensor(edge_detector)
kernel = torch.nn.Parameter( kernel ) # calling this turns tensor into "weight" parameter
m.weight = kernel

print(m.weight)

imgs = sr_utils.load_LR_HR_imgs_sr("images/zebra_GT.fits", (WIDTH, HEIGHT), (119, 326), 4, dtype)

in_tensor = imgs['HR_torch']

output = m(in_tensor).detach().cpu()

hdu = fits.PrimaryHDU(output[0][0])
hdu.writeto('zebra_edges_{}.fits'.format(KERNEL_SIZE))




