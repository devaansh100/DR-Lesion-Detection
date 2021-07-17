import torch
import yaml

def read_config():
	config_file = open('/Users/devaanshgupta/Desktop/PS-I/DR-Lesion-Detection/src/config/config_default.yaml', 'r')
	config = yaml.safe_load(config_file)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config['DEVICE'] = DEVICE
	return config