import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class StandardCNNBlock:
	def __init__(self, kernel_size, in_channels, out_channels):
		'''
		kernel_size: the size of the kernel for the convolution
		in_channels: the number of input channels
		out_channels: the number of filters to be used in the convolution
		'''

class SqueezeAndExcitation(nn.Module):
	def __init__(self, kernel_size, in_features, reduced_features):
		'''
		kernel_size: the size of the input image/size of the kernel for global average pooling
		in_features: the number of channels in the input cube
		reduced_features: the number of features in the squeezed representation
		'''
		super(SqueezeAndExcitation, self).__init__()
		self.fc1 = nn.AvgPool2d(kernel_size = kernel_size)
		self.squeeze = nn.Linear(
						in_features = in_features,
						out_features = reduced_features
					)
		self.excite = nn.Linear(
						in_features = reduced_features,
						out_features = in_features
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
						kernel_size  = 1
					)
		self.conv2dw = nn.Conv2d(
						in_channels = expand_channels,
						out_channels = expand_channels,
						kernel_size = kernel_size,
						stride = stride,
						groups = expand_channels
					)
		self.se = SqueezeAndExcitation(
						in_resolution = in_resolution,
						in_features = expand_channels,
						reduced_features = squeeze_channels
					)
		self.conv3 = nn.Conv2d(
						in_channels = expand_channels,
						out_channels = out_channels,
						kernel_size = 1
					)
		self.batch_norm = nn.BatchNorm2d(
						num_features = out_channels
					)
		self.use_skip = in_channels == out_channels and stride == 1
		self.survival_threshold = 0.7

	def forward(self, x):

		initial = x
		x = self.conv1(initial)
		x = self.conv2dw(x)
		x = F.relu(x)
		x = x * self.se.forward(x)
		x = F.silu(x)
		x = self.conv3(x)
		x = self.batch_norm(x)
		with_skip_connection = stochastic_depth(initial, x)
		return with_skip_connection

	def stochastic_depth(self, inital, x):

		if self.training and self.use_skip:
			survival = torch.rand(x.shape[0], 1, 1 , 1) < self.survival_threshold
			x = torch.div(x, self.survival_threshold) * survival
			return x + initial
		else:	
			if self.use_skip:
				return x + initial
			else:
				return x


	def extract_saliency_map(self, layer):
		pass

testNet = InverseResidualBlock(3,3,3,3,1,4)
print(testNet)