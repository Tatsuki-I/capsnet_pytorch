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


class Decoder(nn.Module):
	def __init__(self, channels, image_width, image_height):
		super(Decoder, self).__init__()

		self.in_vector_size = 160 # digit_capsules x digit_capsule_size = 10 x 16

		self.out_image_channels = channels
		self.out_image_width = image_width
		self.out_image_height = image_height

		out_size = self.out_image_width * self.out_image_height * self.out_image_channels

		self.linear0 = nn.Linear(in_features=self.in_vector_size, out_features=512)
		self.linear1 = nn.Linear(in_features=512, out_features=1024)
		self.linear2 = nn.Linear(in_features=1024, out_features=out_size)

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x: [batch_size, 160]

		h = self.relu(self.linear0(x))
		h = self.relu(self.linear1(h))
		h = self.sigmoid(self.linear2(h))
		h = h.view(-1, self.out_image_channels, self.out_image_height, self.out_image_width)
		# h: [batch_size, 1, 28, 28]

		return h
