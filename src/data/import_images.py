import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../config/')
from config import read_config

config = read_config()
def get_images():
	'''Gets image names from the images directory'''

	os.chdir(config['DATASET'])
	images = os.listdir() #starting from 1 to avoid the .DS_Store
	return images

def get_images_distribution():
	'''Gets image names of images to augment with the passed label'''
	file = open(config['TRAIN_LABELS'])
	entries = file.read().split('\n')
	entries = [entry.split(' ') for entry in entries]
	labels = [entries[x][1] for x in range(0, len(entries)-1)]
	nImages,_,_ = plt.hist(labels, 5, facecolor='blue', alpha = 0.5)

	images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
	for i in range(1, len(entries) - 1):
		label = int(entries[i][1])
		images[label].append(entries[i][0])

	return images
