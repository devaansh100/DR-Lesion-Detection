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

def preprocess(image):
	'''Calls all the pre-processing steps'''

	#Read image into a numpy array
	try:
		img = cv2.imread(image, -1)
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
			img = cv2.imread(image)

		else:
			print(f'{image} is None')
	except:
		print(f'Writing error in image {image}')

	return 1

def main():
	dataPath = '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/'
	'''Creates a thread for each image'''
	with concurrent.futures.ThreadPoolExecutor() as executor:
		
		# Pre-processing all training images
		images = get_images()
		results = list(tqdm(executor.map(preprocess, images), total=len(images)))

		'''
		Label distribution  - [25809, 2443, 5292, 873, 708]
		Distribution in training before augmentation - [20648,  1955,  4234,   699,   567]
		Distruibution of nAugmentations - [0, 1, 0, 5, 6]
		Distribution in training after augmentation - [20648,  3910,  4234,  4194,  3969]
		Distribution in validation - [5161,  488, 1058,  174,  141]
		'''
		
		#Getting image distribution
		imageDistribution = get_images_distribution()

		#Dividing dataset into training and validation set
		trainingData, validationData = divide_dataset(imageDistribution, [20648, 1955, 4234, 699, 567], [5161, 488, 1058, 174, 141])
		
		#Augmenting images in training dataset
		nAugmentations = [0, 1, 0, 5, 6]
		augmented1 = list(tqdm(executor.map(augment, trainingData[1], [nAugmentations[1] for _ in range(0, len(trainingData[1]))]), total=len(trainingData[1])))
		augmented3 = list(tqdm(executor.map(augment, trainingData[3], [nAugmentations[3] for _ in range(0, len(trainingData[3]))]), total=len(trainingData[3]))) 
		augmented4 = list(tqdm(executor.map(augment, trainingData[4], [nAugmentations[4] for _ in range(0, len(trainingData[4]))]), total=len(trainingData[4]))) 
		
		# Adding the new augmented images into the training data distribution
		for label in trainingData:
			if label == 1 or label == 3 or label == 4:
				length = len(trainingData[label])
				for i in range(0, length):
					for j in range(0, nAugmentations[label]):
						trainingData[label].append(trainingData[label][i] + '-' + str(j+1))


		#Saving the labels in training.csv
		with open(dataPath + 'trainingLabels.csv','w') as training:
			trainingWriter = csv.writer(training)
			trainingWriter.writerow(['img-code', 'class'])
			for label in trainingData:
				for image in trainingData[label]:
					trainingWriter.writerow([image, label])

		#Saving the labels in training.csv
		with open(dataPath + 'validationLabels.csv','w') as validation:
			validationWriter = csv.writer(validation)
			validationWriter.writerow(['img-code', 'class'])
			for label in validationData:
				for image in validationData[label]:
					validationWriter.writerow([image, label])

		# #Moving the files into separate test and validation folders
		for label in validationData:
			for image in validationData[label]:
				try:
					os.rename(dataPath + 'train_data_unzip/train_preprocessed/'+image + '.jpeg', dataPath + 'train_data_unzip/validation_preprocessed/'+image+'.jpeg')
				except:
					print(f'Issue in {image}')
		
if __name__ == '__main__':
	main()