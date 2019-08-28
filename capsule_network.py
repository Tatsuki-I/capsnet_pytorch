#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from conv_layer import Conv1
from primary_caps import PrimaryCaps
from digit_caps import DigitCaps
from decoder import Decoder


class CapsuleNetwork(nn.Module):
	def __init__(self, routing_iters=3, reconstruct=True, gpu=0, channels=1, image_width=28, image_height=28, primary_capsules=1152):
		super(CapsuleNetwork, self).__init__()

		self.gpu = gpu
		self.has_reconstruction = reconstruct

		# Build modules for CapsNet.

		## Convolution layer
		self.conv1 = Conv1(channels=channels)

		## PrimaryCaps layer
		self.primary_caps = PrimaryCaps()

		## DigitCaps layer
		self.digit_caps = DigitCaps(routing_iters=routing_iters, gpu=gpu, primary_capsules=primary_capsules)

		## Decoder for reconstruction
		if reconstruct:
			self.decoder = Decoder(channels=channels, image_width=image_width, image_height=image_height)

	def forward(self, x):
		# x: [bacch_size, 1, 28, 28]

		h = self.conv1(x)
		# h: [batch_size, 256, 20, 20]

		h = self.primary_caps(h)
		# h: [batch_size, 1152=primary_capsules, 8=primary_capsule_size]

		h = self.digit_caps(h)
		# h: [batch_size, 10=digit_capsule, 16=digit_capsule_size]

		return h

	def loss(self, images, input, target, size_average=True):
		# images: [batch_size, 1, 28, 28]
		# input: [batch_size, 10, 16, 1]
		# target: [batch_size, 10]

		margin_loss = self.margin_loss(input, target, size_average)

		if self.has_reconstruction:
			reconstruction_loss = self.reconstruction_loss(images, input, size_average)
		else:
			reconstruction_loss = Variable(torch.zeros(1))
			if self.gpu >= 0:
				reconstruction_loss = reconstruction_loss.cuda(self.gpu)

		loss = margin_loss + reconstruction_loss

		return loss, margin_loss, reconstruction_loss

	def margin_loss(self, input, target, size_average=True):
		# images: [batch_size, 1, 28, 28]
		# input: [batch_size, 10, 16]
		# target: [batch_size, 10]

		batch_size = input.size(0)

		# ||vc|| from the paper.
		v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
		# v_mag: [batch_size, 10, 1]

		# Calculate left and right max() terms from Eq.4 in the paper.
		zero = Variable(torch.zeros(1))
		if self.gpu >= 0:
			zero = zero.cuda(self.gpu)
		m_plus = 0.9
		m_minus = 0.1
		max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
		max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
		# max_l, max_r: [batch_size, 10]

		# This is Eq.4 from the paper.
		loss_lambda = 0.5
		T_c = target
		# T_c: [batch_size, 10]
		L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
		# L_c: [batch_size, 10]
		L_c = L_c.sum(dim=1)
		# L_c: [batch_size]

		if size_average:
			L_c = L_c.mean() # average over batch.
		else:
			L_c = L_c.sum() # sum over batch.

		return L_c

	def reconstruction_loss(self, images, input, size_average=True):
		# images: [batch_size, 1, 28, 28]
		# input: [batch_size, 10, 16]

		batch_size = images.size(0)

		# Reconstruct input image.
		reconstructed = self.reconstruct(input)
		# reconstructed: [batch_size, 1, 28, 28]

		# The reconstruction loss is the sum squared difference between the input image and reconstructed image.
		# Multiplied by a small number so it doesn't dominate the margin (class) loss.
		error = (reconstructed - images).view(batch_size, -1)
		error = error**2
		# error: [batch_size, 784=1*28*28]
		error = torch.sum(error, dim=1)
		# error: [batch_size]

		if size_average:
			error = error.mean() # average over batch.
		else:
			error = error.sum() # sum over batch.

		rec_loss_weight = 0.0005
		error *= rec_loss_weight

		return error

	def reconstruct(self, input):
		# input: [batch_size, 10, 16]

		assert self.has_reconstruction, 'Reconstruction path is disabled. For the reconstruction, configure `reconstruct=True` of CapsuleNetwork.'

		# Get the lengths of capsule outputs.
		v_mag = torch.sqrt((input**2).sum(dim=2))
		# v_mag: [batch_size, 10]

		# Get index of longest capsule output.
		_, v_max_index = v_mag.max(dim=1)
		v_max_index = v_max_index.data
		# v_max_index: [batch_size]

		# Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
		batch_size = input.size(0)
		all_masked = [None] * batch_size
		for batch_idx in range(batch_size):
			# Get one sample from the batch.
			input_batch = input[batch_idx]
			# input_bacth: [10, 16]

			# Copy only the maximum capsule index from this batch sample.
			# This masks out (leaves as zero) the other capsules in this sample.
			batch_masked = Variable(torch.zeros(input_batch.size()))
			if self.gpu >= 0:
				batch_masked = batch_masked.cuda(self.gpu)
			batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
			# batch_masked: [10, 16]

			all_masked[batch_idx] = batch_masked
		# all_masked: [10, 16] * batch_size

		# Stack masked capsules over the batch dimension.
		masked = torch.stack(all_masked, dim=0)
		# masked: [batch_size, 10, 16]
		masked = masked.view(batch_size, -1)
		# masked: [batch_size, 160]

		# Reconstruct input image.
		reconstructed = self.decoder(masked)
		# reconstructed: [batch_size, 1, 28, 28]

		return reconstructed
