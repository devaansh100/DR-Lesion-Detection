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
import sys
sys.path.append('../config/')
from config import read_config
import pandas as pd
from collection import Counter

def main():
	
	config = read_config()
	validation = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.ToTensor())
	loader = DataLoader(dataset = validation, batch_size = 1, shuffle = True, num_workers = config['NUM_WORKERS'])
	correct = 0
	total = validation.__len__()

	model = EfficientNet(0.65, 1)
	model.load_state_dict(torch.load(config['MODEL_PATH'] + config['MODEL_NAME'], map_location = config['DEVICE']))

	label_dist = Counter(validation.labels.iloc[:,1])
	predict_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
	model.eval()
	for batch_idx, (images, labels) in enumerate(tqdm(loader)):
		images = images.to(config['DEVICE'])
		labels = labels.to(config['DEVICE'])

		n_correct, prediction_classes = predict(model(images), labels)
		correct += n_correct
		for x in prediction_classes:
			predict_dist[int(x)] += 1


	print(f'Accuracy: {correct/total}')
	print(f'Labels: {label_dist} \t Predictions: {predict_dist}')

def predict(predictions, labels):
	predictions[predictions < 0.5] = 0
	predictions[torch.logical_and(predictions >= 0.5, predictions < 1.5)] = 1
	predictions[torch.logical_and(predictions >= 1.5, predictions < 2.5)] = 2
	predictions[torch.logical_and(predictions >= 2.5, predictions < 3.5)] = 3
	predictions[predictions >= 3.5] = 4

	n_correct_predictions = (predictions == labels).sum()

	return n_correct_predictions, predictions


if __name__ == '__main__':
	main()


