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

	validation = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.ToTensor())
	loader = DataLoader(dataset = validation, batch_size = 1, shuffle = True, num_workers = config['NUM_WORKERS'])
	correct = 0
	total = validation.__len__()

	model = EfficientNet(0.65, 1)
	model.load_state_dict(torch.load(config['MODEL_PATH'] + config['MODEL_NAME'], map_location = DEVICE))
	model.eval()
	for batch_idx, (images, labels) in enumerate(tqdm(loader)):
		images = images.to(DEVICE)
		labels = labels.to(DEVICE)

		correct += predict(model(images), labels)

	print(f"Accuracy: {correct/total}")

def predict(features, labels):
	prediction_probabilities = F.softmax(features, dim = -1)
	predicted_classes = torch.from_numpy(np.array([int(torch.argmax(x)) for x in prediction_probabilities]))
	predicted_classes = predicted_classes.to(DEVICE)
	nCorrect_predictions = (predicted_classes == labels).sum()

	return nCorrect_predictions


if __name__ == '__main__':
	main()


