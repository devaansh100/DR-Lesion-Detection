import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset.py import DRDataset
from tqdm import tqdm
from efficientnet import EfficientNet
import numpy as np

NUM_EPOCHS = 100
NUM_WORKERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_IMG = ''
VAL_IMG = ''

training = DRDataset('trainingLabels.csv', '/Volumes/Seagate Backup Drive/DR Kaggle Dataset/train_unzip/train/', transforms.ToTensor())
validation = DRDataset('validationLabels.csv', '/Volumes/Seagate Backup Drive/DR Kaggle Dataset/train_unzip/valid/', transforms.ToTensor())

training_loader = DataLoader(dataset = training, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
validation_loader = DataLoader(dataset = validation, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

model = EfficientNet(0.2, 0)
loss_fn = nn.LogSoftmax()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

min_valid_loss = np.inf

for epoch in range(NUM_EPOCHS):
	
	train_loss = 0.0
	model.train()
	for bacth_idx, (images, labels) in enumerate(tqdm(training_loader)):
		images = images.to(DEVICE)
		labels = labels.to(DEVICE)

		predictions = model(images)
		loss = loss_fn(images, labels)

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
		loss = loss_fn(images, labels)

		valid_loss += loss.item() * images.size(0)

	print(f'Epoch {epoch}: \t Training Loss: {train_loss/len(training_loader)} \t Validation Loss: {valid_loss/len(validation_loader)}')

	if min_valid_loss > valid_loss:
		print(f'Validation Loss decreased from {min_valid_loss:.6f} to {valid_loss:.6f} \t Saving Model')
		torch.save(model.dicts(), 'model.pth')




if __name__ == '__main__':
	main()