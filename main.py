#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from capsule_network import CapsuleNetwork


# Get training parameter settings.

parser = argparse.ArgumentParser(description='CapsNet for MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
					help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
					help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay-factor', type=float, default=0.9, metavar='DF',
					help='factor to decay learning rate (default: 0.9)')
parser.add_argument('--lr-decay-epoch', type=int, default=1, metavar='DE',
					help='how many epochs to wait before decaying learning rate (default: 1)')
parser.add_argument('--routing', type=int, default=3, metavar='R',
					help='iteration numbers for dymanic routing b/w capsules (default: 3)')
parser.add_argument('--no-reconstruct', dest='reconstruct', action='store_false', 
					help='Disable reconstruction loss (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--tb-log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before saving training status to TensorBoard (default: 10)')
parser.add_argument('--tb-image-interval', type=int, default=100, metavar='N',
					help='how many batches to wait before saving reconstructed images to TensorBoard (default: 100)')
parser.add_argument('--log-dir', '-o', default=None, metavar='LD',
					help='directory under `runs` to output TensorBoard event file, reconstructed.png, and original.png (default: <DATETIME>)')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
					help='id of the GPU to use (default: 0)'
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
					help='name of the Dataset to use (default: 0)')

args = parser.parse_args()


# Check CUDA availability.
if args.gpu >= 0:
	assert torch.cuda.is_available(), \
		'Aborted. CUDA does not seem to be available. Use `--gpu -1` option to train with CPUs.'


# Setup TensorBoardX summary writer.
from tensorboardX import SummaryWriter
from datetime import datetime
import os

log_dir = args.log_dir if (args.log_dir is not None) else datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', log_dir)
writer = SummaryWriter(log_dir=log_dir)


# Initialize the random seed.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# Setup data loaders for train/test data.
if args.dataset = 'cifar10':
	train_dataset = datasets.CIFAR10(
		'data', train=True, download=True, 
		transform=transforms.Compose([
			transforms.RandomCrop(32, padding=4, size=(32, 32)), # data augmentation
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	)

	test_dataset = datasets.CIFAR10(
		'data', train=False, download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	)
else:
	train_dataset = datasets.MNIST(
		'data', train=True, download=True, 
		transform=transforms.Compose([
			transforms.RandomCrop(padding=2, size=(28, 28)), # data augmentation
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])
	)

	test_dataset = datasets.MNIST(
		'data', train=False, download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])
	)

kwargs = {'num_workers': 1, 'pin_memory': True} if (args.gpu >= 0) else {}

train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=args.batch_size, 
	shuffle=True, 
	**kwargs
)

test_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size=args.test_batch_size, 
	shuffle=True, 
	**kwargs
)


# Build CapsNet.
model = CapsuleNetwork(routing_iters=args.routing, reconstruct=args.reconstruct, gpu=args.gpu)
if args.gpu >=0:
	model = model.cuda(args.gpu)

print(model)


# Setup optimizer.
optimizer = optim.Adam(model.parameters(), lr=args.lr)


if args.reconstruct:
	# Get some random test images for reconstruction testing.
	test_iter = iter(test_loader)
	reconstruction_samples, _ = test_iter.next()

	vutils.save_image(reconstruction_samples, os.path.join(log_dir, 'original.png'), normalize=True)
	writer.add_image('original', vutils.make_grid(reconstruction_samples, normalize=True))

	reconstruction_samples = Variable(reconstruction_samples)
	if args.gpu >= 0:
		reconstruction_samples = reconstruction_samples.cuda(args.gpu)


# Function to reconstruct the test images.
def reconstruct_test_images():
	model.eval()

	with torch.no_grad():
		output = model(reconstruction_samples)

	reconstructed = model.reconstruct(output)
	reconstructed = reconstructed.data.cpu()

	return reconstructed


# Function to convert batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length=10):
	batch_size = x.size(0)
	x_one_hot = torch.zeros(batch_size, length)
	for i in range(batch_size):
		x_one_hot[i, x[i]] = 1.0
	return x_one_hot


# Function to get learning rates from the optimizer.
def get_lr():
	lr_params = []
	for param_group in optimizer.param_groups:
		lr_params.append(param_group['lr'])
	return lr_params


# Function to decay learning rate.
def decay_lr(epoch):
	if epoch % args.lr_decay_epoch != (args.lr_decay_epoch - 1):
		return
	for param_group in optimizer.param_groups:
		param_group['lr'] *= args.lr_decay_factor


# Function for training.
def train(epoch):
	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):
		target_one_hot = to_one_hot(target)

		data, target = Variable(data), Variable(target_one_hot)
		if args.gpu >= 0:
			data, target = data.cuda(args.gpu), target.cuda(args.gpu)

		optimizer.zero_grad()
		output = model(data) # forward.
		loss, margin_loss, reconstruction_loss = model.loss(data, output, target)
		loss.backward()
		optimizer.step()

		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * args.batch_size, len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item() )
		)
		
		if args.reconstruct:
			reconstructed = reconstruct_test_images()
			vutils.save_image(reconstructed, os.path.join(log_dir, 'reconstructed.png'), normalize=True)
		
		n_iter = epoch * len(train_loader) + batch_idx

		if n_iter % args.tb_log_interval == 0:
			# Log train/loss to TensorBoard.
			writer.add_scalar('train/loss', loss.item(), n_iter)
			writer.add_scalar('train/loss_margin', margin_loss.item(), n_iter)
			if args.reconstruct:
				writer.add_scalar('train/loss_reconstruction', reconstruction_loss.item(), n_iter)

			# Log base learning rate to TensorBoard.
			lr = get_lr()[0]
			writer.add_scalar('lr', lr, n_iter)

		if args.reconstruct and (n_iter % args.tb_image_interval == 0):
			# Log reconstructed test images to TensorBoard.
			writer.add_image(
				'reconstructed/iter_{}'.format(n_iter), 
				vutils.make_grid(reconstructed, normalize=True)
			)

	decay_lr(epoch)


# Function for testing.
def test(epoch):
	model.eval()
	test_loss, test_margin_loss, test_rec_loss = 0., 0., 0.
	correct = 0

	for data, target in test_loader:
		target_indices = target
		target_one_hot = to_one_hot(target_indices)

		data, target = Variable(data), Variable(target_one_hot)
		if args.gpu >= 0:
			data, target = data.cuda(args.gpu), target.cuda(args.gpu)
		
		with torch.no_grad():
			output = model(data)

		# Sum up batch loss by `size_average=False`, later being averaged over all test samples.
		loss, margin_loss, reconstruction_loss = model.loss(data, output, target, size_average=False)
		loss, margin_loss, reconstruction_loss = loss.item(), margin_loss.item(), reconstruction_loss.item()

		test_loss += loss
		test_margin_loss += margin_loss
		test_rec_loss += reconstruction_loss
		
		v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))
		pred = v_mag.data.max(1, keepdim=True)[1].cpu()
		correct += pred.eq(target_indices.view_as(pred)).sum()

	# Average over all test samples.
	test_loss /= len(test_loader.dataset)
	test_margin_loss /= len(test_loader.dataset)
	test_rec_loss /= len(test_loader.dataset)

	test_accuracy = 100. * float(correct) / float(len(test_loader.dataset))
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), test_accuracy )
	)

	# Log test/loss and test/accuracy to TensorBoard at every epoch.
	n_iter = epoch * len(train_loader)
	writer.add_scalar('test/loss', test_loss, n_iter)
	writer.add_scalar('test/loss_margin', test_margin_loss, n_iter)
	if args.reconstruct:
		writer.add_scalar('test/loss_reconstruction', test_rec_loss, n_iter)
	writer.add_scalar('test/accuracy', test_accuracy, n_iter)


# Start training.
for epoch in range(args.epochs):
	train(epoch)
	test(epoch)

# Close TensorBoardX summary writer.
writer.close()
