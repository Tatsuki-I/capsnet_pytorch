#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


class Conv1(nn.Module):
	def __init__(self, channels):
		super(Conv1, self).__init__()

		self.conv = nn.Conv2d(
			in_channels=channels,
			out_channels=256,
			kernel_size=9,
			stride=1,
			bias=True
		)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# x: [batch_size, 1, 28, 28]

		h = self.relu(self.conv(x))
		# h: [batch_size, 256, 20, 20]

		return h
