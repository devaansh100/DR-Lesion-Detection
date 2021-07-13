import cv2
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
import random
import csv
from import_images import get_images, get_images_distribution
from data_augmentation import augment, divide_dataset
from preprocessing import mask, crop_edges
import sys
sys.path.append('../config/')
from config import read_config

config = read_config()

def preprocess(image):
	'''Calls all the pre-processing steps'''
	#Read image into a numpy array
	try:
		img = cv2.imread(config['DATASET'] + image, -1)
		if img is None:
			return 0
	except:
		print(f'Reading error in image {image}')
		return 0

	#Cropping the extra edges
	img = crop_edges(img)

	#Resize the image and make the background white
	img = mask(img)

	#Add condition for image of size zero

	#save image
	try:
		if img is not None:
			img = cv2.imwrite(config['TEST_IMG'] + image, img)

		else:
			print(f'{image} is None')
	except:
		print(f'Writing error in image {image}')

	return 1

def main():
	'''Creates a thread for each image'''
	with concurrent.futures.ThreadPoolExecutor() as executor:
		
		# Pre-processing all training images
		# images = get_images()
		# results = list(tqdm(executor.map(preprocess, images), total=len(images)))

		'''
		Kaggle Dataset

		Label distribution  - [25809, 2443, 5292, 873, 708]
		Distribution in training before augmentation - [20648,  1955,  4234,   699,   567]
		Distruibution of nAugmentations - [0, 1, 0, 5, 6]
		Distribution in training after augmentation - [20648,  3910,  4234,  4194,  3969]
		Distribution in validation - [5161,  488, 1058,  174,  141]
		'''

		'''
		DDR Dataset

		Distribution in training - [3133, 2238, 118, 456, 575]
		'''
		
		#Getting image distribution
		training_data = get_images_distribution()

		#Dividing dataset into training and validation set
		training_data, validation_data = divide_dataset(image_distribution, [20648, 1955, 4234, 699, 567], [5161, 488, 1058, 174, 141])
		
		#Augmenting images in training dataset
		nAugmentations = [0, 1, 0, 5, 6]
		augmented1 = list(tqdm(executor.map(augment, training_data[1], [nAugmentations[1] for _ in range(0, len(training_data[1]))]), total=len(training_data[1])))
		augmented3 = list(tqdm(executor.map(augment, training_data[3], [nAugmentations[3] for _ in range(0, len(training_data[3]))]), total=len(training_data[3]))) 
		augmented4 = list(tqdm(executor.map(augment, training_data[4], [nAugmentations[4] for _ in range(0, len(training_data[4]))]), total=len(training_data[4]))) 
		
		# Adding the new augmented images into the training data distribution
		for label in training_data:
			label = int(label)
			if label == 1 or label == 3 or label == 4:
				length = len(training_data[label])
				for i in range(0, length):
					for j in range(0, nAugmentations[label]):
						training_data[label].append(training_data[label][i] + '-' + str(j+1))

		#Saving the labels in trainingLabels.csv
		with open(trainLabels,'w') as training:
			training_writer = csv.writer(training)
			training_writer.writerow(['img-code', 'class'])
			for label in training_data:
				for image in training_data[label]:
					training_writer.writerow([image, label])

		#Saving the labels in validationLabels.csv
		with open(config['VAL_LABELS'],'w') as validation:
			validation_writer = csv.writer(validation)
			validation_writer.writerow(['img-code', 'class'])
			for label in validation_data:
				for image in validation_data[label]:
					validation_writer.writerow([image, label])

		#Moving the files into separate test and validation folders
		for label in validation_data:
			for image in validation_data[label]:
				try:
					os.rename(config['TRAIN_IMG'] + image + '.jpeg', config['VAL_IMG'] + image + '.jpeg')
				except:
					print(f'Issue in {image}')
		
if __name__ == '__main__':
	main()