import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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