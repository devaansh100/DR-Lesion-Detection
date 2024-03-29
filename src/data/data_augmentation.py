import random
import cv2
import numpy as np
import sys
sys.path.append('../config/')
from config import read_config

config = read_config()

def augment(image, nAugmentations):
	'''Performs data augmentation on each image provided in a list'''
	try:
		img = cv2.imread(config['DATASET'] + image + '.jpeg')
		if img is None:
			print(f'{image} is of type None')
			return 0
	except:
		return 0

	for i in range(nAugmentations):

		option = random.randint(1,4)
		new_image = img
		

		#Randomly choose to flip horizontally, vertically, or both
		if option == 1:
			flip_code = random.randint(-1, 1)
			new_image = cv2.flip(img, flip_code)

		#Rotate the image by a random angle
		elif option == 2:
			angle = random.random()*360
			row,col,_ = img.shape
			center = tuple(np.array([row,col])/2)
			rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
			new_image_0 = cv2.warpAffine(img[:,:,0], rot_mat, (col,row))
			new_image_1 = cv2.warpAffine(img[:,:,1], rot_mat, (col,row))
			new_image_2 = cv2.warpAffine(img[:,:,2], rot_mat, (col,row))
			new_image = np.stack((new_image_0, new_image_1, new_image_2), axis = -1)

		#Enhance the image by the CLAHE operator 
		elif option == 3:
			clahe = cv2.createCLAHE(clipLimit = 3)
			new_image_r = clahe.apply(img[:,:,0])
			new_image_g = clahe.apply(img[:,:,1])
			new_image_n = clahe.apply(img[:,:,2])

			new_image_ = np.stack((new_image_r, new_image_g, new_image_b), axis = -1)

		#Add noise
		elif option == 4:
			noises = ['gaussian', 'poisson', 'speckle', 's&p']
			noise = random.choice(noises)
			new_image = random_noise(img, mode = noise, seed = None, clip = True)
			new_image = new_image*255

		try:
			#Saving the augmented image
			cv2.imwrite(config['TRAIN_IMG'] + image + '-'+str(i+1)+'.jpeg', new_image)
			# cv2.imwrite(image.replace('.jpeg','') + '-'+str(i+1)+'.jpeg', new_image)

		except:
			continue
	return 1

def divide_dataset(dist, training_dist, validation_dist):
	'''Divides the input distribution into a training and validation dataset'''

	ntraining = training_dist
	nValidation = validation_dist

	training = {0: [], 1: [], 2: [], 3: [], 4: []}
	validation = {0: [], 1: [], 2: [], 3: [], 4: []}

	for label in dist:
		random.shuffle(dist[label])
		training[label] = dist[label][0:ntraining[label]]
		validation[label] = dist[label][-nValidation[label]:]

	return training, validation