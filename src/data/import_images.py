import os
import matplotlib.pyplot as plt

data_path = '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/'
def get_images():
	'''Gets image names from the images directory'''

	os.chdir(data_path + 'train_data_unzip/train/')
	images = os.listdir() #starting from 1 to avoid the .DS_Store
	return images

def get_images_distribution():
	'''Gets image names of images to augment with the passed label'''
	file = open(data_path + 'trainLabels.csv')
	entries = file.read().split('\n')
	entries = [entry.split(',') for entry in entries]
	labels = [entries[x][1] for x in range(1, len(entries)-1)]
	nImages,_,_ = plt.hist(labels, 5, facecolor='blue', alpha = 0.5)

	images = {0: [], 1: [], 2: [], 3: [], 4: []}
	for i in range(1, len(entries) - 1):
		label = int(entries[i][1])
		images[label].append(entries[i][0])

	return images
