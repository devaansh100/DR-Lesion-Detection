import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from inverse_residual_block import InverseResidualBlock
from squeeze_excitation import SqueezeAndExcitation
from standard_cnn import StandardCNNBlock

testNet = InverseResidualBlock(3,3,3,3,1,4)
print(testNet)