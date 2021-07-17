import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

googlenet = models.googlenet(pretrained=True)

for param in googlenet.parameters():
	param.requires_grad = False
# print(model)
class Pretrained(nn.Module):
	def __init__(self, pretrained_model):
		super(Pretrained, self).__init__()
		self.pretrained = pretrained_model
		self.fc2 = nn.Linear(1000, 100)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(100, 20)
		self.classifier = nn.Linear(20, 6)
	  
	def forward(self, x):
		x = self.pretrained(x)
		x = self.fc2(x)
		x = self.relu2(x)
		x = self.fc3(x)
		x = self.classifier(x)
		return x

model = Pretrained(googlenet)