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

def main():
	NUM_EPOCHS = 100
	NUM_WORKERS = 2
	BATCH_SIZE = 32
	LEARNING_RATE = 1e-3
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	TRAIN_IMG = '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train_preprocessed/'
	VAL_IMG = '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/validation_preprocessed/'
	MODEL_PATH = '/Users/devaanshgupta/Desktop/PS-I/DR-Lesion-Detection/models/'

	training = DRDataset('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/trainingLabels.csv', TRAIN_IMG, transforms.ToTensor())
	validation = DRDataset('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/validationLabels.csv', VAL_IMG, transforms.ToTensor())

	training_loader = DataLoader(dataset = training, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
	validation_loader = DataLoader(dataset = validation, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

	model = EfficientNet(0.2, 0)
	model = model.to(DEVICE)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

	min_valid_loss = np.inf

	for epoch in range(NUM_EPOCHS):
		
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
			predictions = predictions.view(BATCH_SIZE, 5)
			labels = labels.view(BATCH_SIZE)

			loss = loss_fn(predictions, labels)

			valid_loss += loss.item() * images.size(0)

		print(f'Epoch {epoch}: \t Training Loss: {train_loss/len(training_loader)} \t Validation Loss: {valid_loss/len(validation_loader)}')

		if min_valid_loss > valid_loss:
			print(f'Validation Loss decreased from {min_valid_loss:.6f} to {valid_loss:.6f} \t Saving Model')
			torch.save(model.state_dict(), MODEL_PATH + 'model.pth')
			min_valid_loss = valid_loss
if __name__ == '__main__':
	main()
