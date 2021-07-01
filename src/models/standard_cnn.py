import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class StandardCNNBlock(nn.Module):
	def __init__(self, kernel_size, in_channels, out_channels, stride, padding):
		'''
		kernel_size: the size of the kernel for the convolution
		in_channels: the number of input channels
		out_channels: the number of filters to be used in the convolution
		'''
		super(StandardCNNBlock, self).__init__()
		self.conv1 = nn.Conv2d(
						in_channels = in_channels,
						kernel_size = kernel_size,
						out_channels = out_channels,
						stride = stride,
						padding = padding,
						bias = False
					)
		self.batch_norm = nn.BatchNorm2d(
						num_features = out_channels
					)

	def forward(self, x):

		x = self.conv1(x)
		x = self.batch_norm(x)
		x = F.silu(x)

		return x
