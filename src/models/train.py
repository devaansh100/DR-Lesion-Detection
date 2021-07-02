import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import DRDataset
from tqdm import tqdm
from efficientnet import EfficientNet
import numpy as np
import torch.nn as nn
import yaml

def main():
	config_file = open('config.yml', 'r')
	config = yaml.safe_load(config_file)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	training = DRDataset('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/trainingLabels.csv', config['TRAIN_IMG'], transforms.ToTensor())
	validation = DRDataset('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/validationLabels.csv', config['VAL_IMG'], transforms.ToTensor())

	training_loader = DataLoader(dataset = training, batch_size = config['BATCH_SIZE'], shuffle = True, num_workers = config['NUM_WORKERS'])
	validation_loader = DataLoader(dataset = validation, batch_size = config['BATCH_SIZE'], shuffle = True, num_workers = config['NUM_WORKERS'])

	model = EfficientNet(0.2, 0)
	model.apply(weights_init)
	model = model.to(DEVICE)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = config['LEARNING_RATE'])

	min_valid_loss = np.inf


	for epoch in range(config['NUM_EPOCHS']):
		
		train_loss = 0.0
		model.train()
		for batch_idx, (images, labels) in enumerate(tqdm(training_loader)):
			images = images.to(DEVICE)
			labels = labels.to(DEVICE)

			predictions = model(images)

			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 5)
			labels = labels.view(batch_number)
			# print(predictions)
			# print(labels)

			loss = loss_fn(predictions, labels)

			optimizer.zero_grad()
			loss.backward()

			optimizer.step()

			train_loss += loss.item() * images.size(0)
			
		valid_loss = 0.0
		model.eval()	
		for batch_idx, (images, labels) in enumerate(tqdm(validation_loader)):
			images = images.to(DEVICE)
			labels = labels.to(DEVICE)

			predictions = model(images)

			batch_number = predictions.shape[0]
			predictions = predictions.view(batch_number, 5)
			labels = labels.view(batch_number)

			loss = loss_fn(predictions, labels)

			valid_loss += loss.item() * images.size(0)

		print(f'Epoch {epoch}: \t Training Loss: {train_loss/len(training_loader)} \t Validation Loss: {valid_loss/len(validation_loader)}')

		if min_valid_loss > valid_loss:
			print(f'Validation Loss decreased from {min_valid_loss:.6f} to {valid_loss:.6f} \t Saving Model')
			torch.save(model.state_dict(), config['MODEL_PATH'] + config['model.pth'])
			min_valid_loss = valid_loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':
	main()
