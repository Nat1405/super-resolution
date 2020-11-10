# Use GPU if available
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    dtype = torch.FloatTensor

i = 0
net = None
imgs = None
history = {
	"psnr_HR":[],
	"psnr_LR":[],
	"target_loss":[],
	"training_loss":[]
}