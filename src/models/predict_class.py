import torch
from efficientnet import EfficientNet
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DRDataset
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import yaml

def main():
	config_file = open('config.yml', 'r')
	config = yaml.safe_load(config_file)

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	validation = DRDataset(config['VALID_LABELS'], config['VAL_IMG'], transforms.ToTensor())
	loader = DataLoader(dataset = validation, batch_size = 1, shuffle = True, num_workers = config['NUM_WORKERS'])
	correct = 0
	total = validation.__len__()

	model = EfficientNet(0.2, 0)
	model.load_state_dict(torch.load(config['MODEL_PATH'] + config['MODEL_NAME'], map_location = DEVICE))
	model.eval()
	for batch_idx, (images, labels) in enumerate(tqdm(loader)):
		images = images.to(DEVICE)
		labels = labels.to(DEVICE)

		predictions = model(images)

		batch_number = predictions.shape[0]
		predictions = predictions.view(batch_number, 5)
		labels = labels.view(batch_number)
		predicted_class = torch.argmax(predictions)

		if predicted_class == labels:
			correct += 1
	print(f"Accuracy: {correct/total}")

if __name__ == '__main__':
	main()


