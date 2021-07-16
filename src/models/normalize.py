import torch
import yaml
import sys
sys.path.append('../config/')
from config import read_config
from tqdm import tqdm

config = read_config()
def find_statistic_param(dataset):
	means_r = []
	means_g = []
	means_b = []

	stds_r = []
	stds_g = []
	stds_b = []
	for i in tqdm(range(0, dataset.__len__())):
		img, _ = dataset.__getitem__(i)
		means_r.append(torch.mean(img[0,:,:]))
		means_g.append(torch.mean(img[1,:,:]))
		means_b.append(torch.mean(img[2,:,:]))

		stds_r.append(torch.std(img[0,:,:]))
		stds_g.append(torch.std(img[1,:,:]))
		stds_b.append(torch.std(img[2,:,:]))

	mean_r = torch.mean(torch.tensor(means_r))
	mean_g = torch.mean(torch.tensor(means_g))
	mean_b = torch.mean(torch.tensor(means_b))

	std_r = torch.std(torch.tensor(stds_r))
	std_g = torch.std(torch.tensor(stds_g))
	std_b = torch.std(torch.tensor(stds_b))

	return ([float(mean_r), float(mean_g), float(mean_b)], [float(std_r), float(std_g), float(std_b)])