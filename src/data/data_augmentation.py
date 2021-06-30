import random
import cv2
import numpy as np

def augment(image, nAugmentations):
	'''Performs data augmentation on each image provided in a list'''
	try:
		img = cv2.imread('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train_preprocessed/' + image + '.jpeg')
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
			flipCode = random.randint(-1, 1)
			new_image = cv2.flip(img, flipCode)

		#Rotate the image by a random angle
		elif option == 2:
			angle = random.random()*360
			row,col,_ = img.shape
			center = tuple(np.array([row,col])/2)
			rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
			new_image0 = cv2.warpAffine(img[:,:,0], rot_mat, (col,row))
			new_image1 = cv2.warpAffine(img[:,:,1], rot_mat, (col,row))
			new_image2 = cv2.warpAffine(img[:,:,2], rot_mat, (col,row))
			new_image = np.stack((new_image0, new_image1, new_image2), axis = -1)

		#Enhance the image by the CLAHE operator 
		elif option == 3:
			clahe = cv2.createCLAHE(clipLimit = 3)
			new_imageR = clahe.apply(img[:,:,0])
			new_imageG = clahe.apply(img[:,:,1])
			new_imageB = clahe.apply(img[:,:,2])

			new_image = np.stack((new_imageR, new_imageG, new_imageB), axis = -1)

		#Add noise
		elif option == 4:
			noises = ['gaussian', 'poisson', 'speckle', 's&p']
			noise = random.choice(noises)
			new_image = random_noise(img, mode = noise, seed = None, clip = True)
			new_image = new_image*255

		try:
			#Saving the augmented image
			cv2.imwrite('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train_preprocessed/' + image + '-'+str(i+1)+'.jpeg', new_image)
			# cv2.imwrite(image.replace('.jpeg','') + '-'+str(i+1)+'.jpeg', new_image)

		except:
			continue
	return 1

def divide_dataset(dist, trainingDist, validationDist):
	'''Divides the input distribution into a training and validation dataset'''

	ntraining = trainingDist
	nValidation = validationDist

	training = {0: [], 1: [], 2: [], 3: [], 4: []}
	validation = {0: [], 1: [], 2: [], 3: [], 4: []}

	for label in dist:
		random.shuffle(dist[label])
		training[label] = dist[label][0:ntraining[label]]
		validation[label] = dist[label][-nValidation[label]:]

	return training, validation