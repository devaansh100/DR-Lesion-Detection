import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation(nn.Module):
	def __init__(self, in_features, reduced_features):
		'''
		in_features: the number of channels in the input cube
		reduced_features: the number of features in the squeezed representation
		'''
		super(SqueezeAndExcitation, self).__init__()
		self.fc1 = nn.AdaptiveAvgPool2d((1, 1))
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