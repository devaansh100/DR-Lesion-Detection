import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from inverted_residual_block import InvertedResidualBlock
from squeeze_excitation import SqueezeAndExcitation
from standard_cnn import StandardCNNBlock

baseline_mobilenet_params = [	
	#expand_ratio, kernel_size, out_channels, stride, layer_repeats
	(1, 3, 16, 1, 1),
	(6, 3, 24, 2, 2),
	(6, 5, 40, 2, 2),
	(6, 3, 80, 2, 3),
	(6, 5, 112, 1, 3),
	(6, 5, 192, 2, 4),
	(6, 3, 320, 1, 1),
]

class EfficientNet(nn.Module):
	def __init__(self, dropout_rate, phi, alpha = 1.2, beta = 1.1, gamma = 1.15):
		super(EfficientNet, self).__init__()
		self.depth_factor, self.width_factor, self.resolution_factor = self.calc_expand_factors(alpha, beta, gamma, phi)
		self.input_size = ceil(224*self.resolution_factor)

		initial_layers = self.define_initial_layers()
		mobilenet_layers = self.define_mobilenet_layers()
		classifier =  self.define_classfier(dropout_rate)

		self.network = initial_layers + mobilenet_layers + classifier
		self.seq = nn.Sequential(*self.network)
		

	def calc_expand_factors(self, alpha, beta, gamma, phi):
		depth_factor = alpha**phi
		width_factor = beta**phi
		resolution_factor = beta**phi
		return depth_factor, width_factor, resolution_factor

	def define_initial_layers(self):
		layers = []
		in_channels = 3
		out_channels = ceil(32*self.width_factor)
		for layer in range(1*ceil(self.depth_factor)):
			layers.append(
						StandardCNNBlock(
								in_channels = in_channels,
								out_channels = out_channels,
								kernel_size = 3,
								padding = 1,
								stride = 2 if layer == 0 else 1
							)	
						)
			in_channels = out_channels

		return layers

	def define_mobilenet_layers(self):
		layers = []
		in_channels = ceil(32*self.width_factor)
		for expand_ratio, kernel_size, out_channels, stride, layer_repeats in baseline_mobilenet_params:
			nLayers = ceil(self.depth_factor*layer_repeats)
			out_channels = 4*ceil(int(out_channels*self.width_factor)/4)

			for layer in range(nLayers):
				layers.append(
							InvertedResidualBlock(
										in_channels = in_channels,
										expand_channels = expand_ratio*in_channels,
										kernel_size = kernel_size,
										out_channels = out_channels,
										stride = stride if layer == 0 else 1,
										padding = kernel_size//2
									)
						)
				in_channels = out_channels

		return layers

	def define_classfier(self, dropout_rate):
		layers = []
		layers.append(
						StandardCNNBlock(
								kernel_size = 1, 
								in_channels = 4*ceil(int(320*self.width_factor)/4), 
								out_channels = ceil(1280*self.width_factor), 
								stride = 1, 
								padding = 0
							)
					)
		layers.append(
						nn.Dropout(
								dropout_rate
							)
					)

		layers.append(
						nn.AdaptiveAvgPool2d(
								output_size = (1,1)
							)
					)
		
		layers.append(
						nn.Conv2d(
								in_channels = ceil(1280*self.width_factor), 
								out_channels = 5,
								kernel_size = 1
							)
					)
		layers.append(
						nn.SiLU()
					)
		return layers

	def forward(self, x):
		return self.seq(x)


# Add check for image resolution
