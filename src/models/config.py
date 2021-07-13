import torch
import yaml

def read_config():
	config_file = open('config.yml', 'r')
	config = yaml.safe_load(config_file)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config['DEVICE'] = DEVICE
	return config