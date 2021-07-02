import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from skimage import io
import torchvision
import torchvision.transforms as transforms

class DRDataset(Dataset):

	def __init__(self, csv_file, root_dir, transform = None):
		labels = pd.read_csv(csv_file, header = None, skiprows = [0])
		self.labels = labels
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.labels)
		pass

	def __getitem__(self, index):
		img_path = self.root_dir + self.labels.iloc[index, 0] + '.jpeg'
		img = io.imread(img_path)
		truth_value = torch.tensor(int(self.labels.iloc[index, 1])) 

		if self.transform:
			img = self.transform(img)

		return (img, truth_value)