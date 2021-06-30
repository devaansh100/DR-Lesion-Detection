import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class SqueezeAndExcitation(nn.Module):
	def __init__(self, kernel_size, in_channels, reduced_channels): #decide default value for ratio
		'''
			kernel_size = the size of the input image/size of the kernel for global average pooling
			in_channels = the number of channels in the input cube
			ratio = the ratio by which the compression is to be applied
		'''
		super(SqueezeAndExcitation, self).__init__()
		self.fc1 = nn.AvgPool2d(kernel_size = kernel_size)
		self.squeeze = nn.Linear(
						in_channels = in_channels,
						out_channels = reduced_channels
					)
		self.excite = nn.Linear(
						in_channels = reduced_channels,
						out_channels = in_channels
					)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.squeeze(x)
		x = F.relu(x)
		x = self.excite(x)
		x = F.sigmoid(x)

		return x

class InverseResidualBlock(nn.Module):
	def __init__(self, in_channels, in_resolution, expand_channels, kernel_size, stride, out_channels, squeeze_channels = 4):
		'''
			in_channels: The number of channels in the input image
			in_resolution: The input size of the image
			expand_channels: The number of 1x1 filters to be used 
			kernel_size: The kernel size for the depthwise convolution
			stride: The stride for the convolution operator
			out_channels: The number of channels in the final image
			squeeze_channels: The number of channels to which the input channels has to be squeezed in the squeeze and excitation block
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
		self.se = SqueezeAndExcitation(in_resolution[0], expand_channels, squeeze_channels)
		self.conv3 = nn.Conv2d(
						in_channels = expand_channels,
						out_channels = out_channels,
						kernel_size = 1
					)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2dw(x)
		x = F.silu(x)
		x = x * self.se.forward(x)
		x = F.silu(x)
		x = self.conv3(x)

		return x

	def extract_saliency_map(self, layer):
		pass

testNet = InverseResidualBlock(3,3,3,3,1,4)
print(testNet)