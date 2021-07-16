import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
sys.path.append('../config/')

from predict_class import predict
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DRDataset
from tqdm import tqdm
from efficientnet import EfficientNet
from config import read_config
from transer_learning import Pretrained

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal_(m.weight)
		try:
			nn.init.zeros_(m.bias)
		except:
			pass

def train():
	'''Training Loop'''
	# Initialising config and training objects
	config = read_config()
	# model = EfficientNet(0.5, 1)
	googlenet = models.googlenet(pretrained=True)
	for param in googlenet.parameters():
	  param.requires_grad = False
	model = Pretrained(googlenet)
	optimizer = optim.Adam(model.parameters(), lr = config['LEARNING_RATE'])
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
	checkpoint = {'epoch': 0, 'min_valid_loss': np.inf}

	if config['LOAD_CHECKPOINT']:
		checkpoint = torch.load(config['CHECKPOINT_PATH'] + config['CHECKPOINT_MODEL'], map_location = config['DEVICE'])
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	else:
	  model.apply(weights_init)

	model = model.to(config['DEVICE'])
	loss_fn = nn.MSELoss()

	# Defining the dataset and the data loader
	transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize(config['IMG_SIZE'])])
	training = DRDataset(config['TRAIN_LABELS'], config['TRAIN_IMG'], transformations)
	validation = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transformations)

	labels = list(training.labels.iloc[:,1])
	hist = torch.tensor(list(Counter(labels).values()))
	class_weights = 1. / hist
	sample_weights = torch.tensor(class_weights[labels])

	sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)

	training_loader = DataLoader(dataset = training, batch_size = config['BATCH_SIZE'], num_workers = config['NUM_WORKERS'], sampler = sampler)
	validation_loader = DataLoader(dataset = validation, batch_size = config['BATCH_SIZE'], shuffle = True, num_workers = config['NUM_WORKERS'])
	
	min_valid_loss = checkpoint['min_valid_loss']

	# Start training
	for epoch in range(config['NUM_EPOCHS']):
		print(f'Epoch {epoch + 1 + checkpoint['epoch']}:')
		train_loss = 0.0
		correct = 0.0
		total = training.__len__()

		#Looping over all training batches
		model.train()
		train_predict_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		train_correct_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		train_label_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		for batch_idx, (images, labels) in enumerate(tqdm(training_loader)):
			
			# Transferring images to the GPU
			images = images.to(config['DEVICE'])
			labels = labels.to(config['DEVICE'])
			
			# Obtaining the model output
			predictions = model(images)

			# Calculating the loss and the predictions
			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 6).float()
			labels = labels.view(batch_number, 6).float()

			n_correct, predicted_classes = predict(predictions, labels)
			correct += n_correct

			loss = loss_fn(predictions, labels)

			# Backpropagating
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Getting the final training loss
			train_loss += loss.item() * batch_number

			# Filling in the prediction distributions and outputs
			for i in range(len(predicted_classes)):
				x = predicted_classes[i]
				train_predict_dist[int(x)] += 1
				if x == labels[i]:
					train_correct_dist[int(x)] += 1
			for x in labels:
				train_label_dist[int(x)] += 1

		train_accuracy = correct*100/total
		scheduler.step()

		# Initialising validation parameters	
		valid_loss = 0.0
		correct = 0
		total = validation.__len__()

		# Looping over the validation data
		model.eval()
		val_predict_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		val_correct_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		val_label_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		for batch_idx, (images, labels) in enumerate(tqdm(validation_loader)):

			# Transferring images to the GPU
			images = images.to(config['DEVICE'])
			labels = labels.to(config['DEVICE'])

			# Obtaining the model output
			predictions = model(images)

			# Calculating the loss and the predictions
			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 6).float()
			labels = labels.view(batch_number, 6).float()

			# Filling in the prediction distributions and outputs
			n_correct, predicted_classes = predict(predictions, labels)
			correct += n_correct

			# Calculating the loss
			loss = loss_fn(predictions, labels)

			valid_loss += loss.item() * images.size(0)

			for i in range(len(predicted_classes)):
				x = predicted_classes[i]
				val_predict_dist[int(x)] += 1
				if x == labels[i]:
					val_correct_dist[int(x)] += 1
			for x in labels:
				val_label_dist[int(x)] += 1

		validation_accuracy = correct*100/total

		# Converting the training and validation predictions to percentages
		for key in val_correct_dist:
			val_correct_dist[key] = 100*val_correct_dist[key]/val_label_dist[key]

		for key in train_correct_dist:
			train_correct_dist[key] = 100*train_correct_dist[key]/train_label_dist[key]

		# rounding the percentages to 4 decimal places
		val_correct_dist = {key: round(val_correct_dist[key],4) for key in val_correct_dist}
		train_correct_dist = {key: round(train_correct_dist[key],4) for key in train_correct_dist}

		# Printing after every epoch
		print(f'Training Loss: {train_loss/len(training_loader)} \t Validation Loss: {valid_loss/len(validation_loader)}')
		print(f'Training Accuracy: {train_accuracy} \t Validation Accuracy: {validation_accuracy}')
		print(f'Training Labels: {train_label_dist} \t Validation Labels: {val_label_dist}')
		print(f'Training Predictions: {train_predict_dist} \t Validation Predictions: {val_predict_dist}')
		print(f'Training Correct Predictions: {train_correct_dist} \t Validation Correct Predictions: {val_correct_dist}')

		# Saving the model if the validation loss decreases
		if min_valid_loss > valid_loss:
			print(f'Validation Loss decreased from {min_valid_loss:.6f} to {valid_loss:.6f} \t Saving Model\n')
			torch.save(model.state_dict(), config['MODEL_PATH'] + config['model.pth'])
			min_valid_loss = valid_loss

		# Saving the checkpoint after every epoch
		torch.save({
		    'epoch': epoch,
		    'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		    'scheduler_state_dict': scheduler.state_dict(),
		    'loss': train_loss,
		    'min_valid_loss': min_valid_loss
		    }, config['CHECKPOINT_PATH'] + config['CHECKPOINT_MODEL'])

if __name__ == '__main__':
	train()
