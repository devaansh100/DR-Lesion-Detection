import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DRDataset
from tqdm import tqdm
from efficientnet import EfficientNet
import numpy as np
import torch.nn as nn
import yaml
import torch.nn.functional as F
import pandas as pd
from predict_class import predict
from collections import Counter

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal_(m.weight)
		try:
			nn.init.zeros_(m.bias)
		except:
			pass

def train():
	config_file = open('config.yml', 'r')
	config = yaml.safe_load(config_file)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = EfficientNet(0.5, 1)
	model.apply(weights_init)
	model = model.to(DEVICE)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = config['LEARNING_RATE'])
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

	training = DRDataset(config['TRAIN_LABELS'], config['TRAIN_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(model.input_size)]))
	validation = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(model.input_size)]))

	training_loader = DataLoader(dataset = training, batch_size = config['BATCH_SIZE'], num_workers = config['NUM_WORKERS'], shuffle = True)
	validation_loader = DataLoader(dataset = validation, batch_size = config['BATCH_SIZE'], shuffle = True, num_workers = config['NUM_WORKERS'])
	min_valid_loss = np.inf

	for epoch in range(config['NUM_EPOCHS']):
		print(f'Epoch {epoch + 1}:')
		train_loss = 0.0
		correct = 0.0
		total = training.__len__()
		model.train()
		for batch_idx, (images, labels) in enumerate(tqdm(training_loader)):
			images = images.to(DEVICE)
			labels = labels.to(DEVICE)
			
			predictions = model(images)

			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 5)
			labels = labels.view(batch_number)

			correct += predict(predictions, labels)

			loss = loss_fn(predictions, labels)
			optimizer.zero_grad()

			loss.backward()

			optimizer.step()

			train_loss += loss.item() * images.size(0)
		train_accuracy = correct*100/total
		scheduler.step()	
		valid_loss = 0.0
		correct = 0
		total = validation.__len__()
		model.eval()	
		for batch_idx, (images, labels) in enumerate(tqdm(validation_loader)):
			images = images.to(DEVICE)
			labels = labels.to(DEVICE)

			predictions = model(images)

			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 5)
			labels = labels.view(batch_number)

			correct += predict(predictions, labels)

			loss = loss_fn(predictions, labels)

			valid_loss += loss.item() * images.size(0)
		validation_accuracy = correct*100/total

		print(f'Training Loss: {train_loss/len(training_loader)} \t Validation Loss: {valid_loss/len(validation_loader)}')
		print(f'Training Accuracy: {train_accuracy} \t Validation Accuracy: {validation_accuracy}')

		if min_valid_loss > valid_loss:
			print(f'Validation Loss decreased from {min_valid_loss:.6f} to {valid_loss:.6f} \t Saving Model')
			torch.save(model.state_dict(), config['MODEL_PATH'] + config['model.pth'])
			min_valid_loss = valid_loss

if __name__ == '__main__':
	train()
