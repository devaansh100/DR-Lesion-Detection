import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil

class SqueezeAndExcitation:
	pass

	

class InverseResidualBlock(nn.Module):
	def __init__(self, in_channels, in_resolution, expand_channels, kernel_size, stride, out_channels):
		'''
			in_channels: The number of channels in the input image
			in_resolution: The input size of the image
			expand_channels: The number of 1x1 filters to be used 
			kernel_size: The kernel size for the depthwise convolution
			stride: The stride for the convolution operator
			out_channels: The number of channels in the final image
		'''
		super(InverseResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(
						in_channels  = in_channels,
						out_channels = expand_channels,
						kernel_size  = 1,
					)
		self.conv2dw = nn.Conv2d(
						in_channels = expand_channels,
						out_channels = expand_channels,
						kernel_size = kernel_size,
						groups = expand_channels
					)
		self.se = SqueezeAndExcitation() # Add code fpr SqueezeAndExcitation then fill params here
		self.conv3 = nn.Conv2d(
						in_channels = expand_channels,
						out_channels = out_channels,
						kernel_size = 1
					)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2dw(x)
		x = F.silu(x)
		x = self.se.forward(x)
		x = F.silu(x)
		x = self.conv3(x)

	def extract_saliency_map(self, layer):
		pass

testNet = InverseResidualBlock(3,3,3,3,1,4)
print(testNet)

					


