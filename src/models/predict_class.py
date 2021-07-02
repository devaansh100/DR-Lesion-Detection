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

def main():
	VAL_IMG = '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/validation_preprocessed/'
	MODEL_PATH = '/Users/devaanshgupta/Desktop/PS-I/DR-Lesion-Detection/models/'
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	BATCH_SIZE = 1
	NUM_WORKERS = 2

	validation = DRDataset('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/validationLabels.csv', VAL_IMG, transforms.ToTensor())
	validation_loader = DataLoader(dataset = validation, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
	correct = 0
	total = validation.__len__()

	model = EfficientNet(0.2, 0)
	model.load_state_dict(torch.load(MODEL_PATH + 'model.pth', map_location = (DEVICE)))
	model.eval()
	for batch_idx, (images, labels) in enumerate(tqdm(validation_loader)):
		images = images.to(DEVICE)
		labels = labels.to(DEVICE)

		predictions = model(images)

		batch_number = predictions.shape[0]
		predictions = predictions.view(BATCH_SIZE, 5)
		labels = labels.view(BATCH_SIZE)
		predicted_class = torch.argmax(predictions)

		if predicted_class == labels:
			correct += 1
	print(f"Accuracy: {correct/total}")

if __name__ == '__main__':
	main()


