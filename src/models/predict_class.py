from efficientnet import EfficientNet
from transfer_learning import Pretrained
from torch.utils.data import DataLoader
from dataset import DRDataset
from tqdm import tqdm
from tqdm import tqdm
from config import read_config
from collection import Counter

import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import sys
sys.path.append('../config/')
import pandas as pd

def main():
	
	config = read_config()
	validation = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.ToTensor())
	loader = DataLoader(dataset = validation, batch_size = 1, shuffle = True, num_workers = config['NUM_WORKERS'])
	correct = 0
	total = validation.__len__()

	# model = EfficientNet(0.65, 1)
	googlenet = models.googlenet(pretrained=True)
	for param in googlenet.parameters():
	  param.requires_grad = False
	model = Pretrained(googlenet)
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

	prediction_probabilities = F.softmax(predictions, dim = -1)
	predicted_classes = torch.from_numpy(np.array([int(torch.argmax(x)) for x in prediction_probabilities]))
	predicted_classes = predicted_classes.to(DEVICE)
	correct += (predicted_classes == labels).sum()
	n_correct_predictions = (predicted_classes == labels).sum()

	return (n_correct_predictions, predicted_classes)

if __name__ == '__main__':
	main()


