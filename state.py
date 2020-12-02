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

class HistoryTracker:
	def __init__(self, augmented_history=False):
		self.iteration = []
		self.psnr_HR = []
		self.psnr_LR = []
		self.target_loss = []
		self.training_loss = []

		if augmented_history:
			self.psnr_blurred = []
			self.psnr_downsampled = []
		else:
			self.psnr_blurred = None
			self.psnr_downsampled = None