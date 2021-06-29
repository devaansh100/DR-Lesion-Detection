import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import concurrent.futures
from imutils import contours
import imutils
from tqdm import tqdm
import random
import csv
from skimage.util import random_noise

def get_images():
	'''Gets image names from the images directory'''

	os.chdir('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train/')
	images = os.listdir() #starting from 1 to avoid the .DS_Store
	return images

def get_images_distribution():
	'''Gets image names of images to augment with the passed label'''
	# file = open('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/trainLabels.csv')
	file = open('/Users/devaanshgupta/Downloads/trainLabels.csv')
	entries = file.read().split('\n')
	entries = [entry.split(',') for entry in entries]
	labels = [entries[x][1] for x in range(1, len(entries)-1)]
	nImages,_,_ = plt.hist(labels, 5, facecolor='blue', alpha = 0.5)

	images = {0: [], 1: [], 2: [], 3: [], 4: []}
	for i in range(1, len(entries) - 1):
		label = int(entries[i][1])
		images[label].append(entries[i][0])

	return images

def mask(img):
	'''Creates a mask for the image to remove outer black borders'''
	if img is None:
		return None
	#Converting from BGR to RGB
	try:
		maskedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	except:
		return None

	#Resizing the image
	maskedImg = cv2.resize(maskedImg, (224,224))

	#Creating the mask. Remove black pixels and change to white pixels
	maskedImg = cv2.addWeighted(maskedImg,4, cv2.GaussianBlur(maskedImg , (0,0) , 25) ,-4 ,128)

	return maskedImg

def crop_edges(img, tol = 7):
	'''Crops the extra edges in the image'''
	
	if img.ndim == 2:
		mask = img > tol
		return img[np.ix_(mask.any(1),mask.any(0))]
	# If we have a normal RGB images
	elif img.ndim == 3:
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		mask = gray_img > tol
		
		check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
		if (check_shape == 0): # image is too dark so that we crop out everything,
			return img # return original image
		else:
			img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
			img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
			img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
			img = np.stack([img1,img2,img3],axis=-1)
		return img

#Compare with the new cropping method
# def remove_extra_edges(img, channel):

# 	rows, cols, _ = img.shape
# 	#Finding the color of the edges
# 	borderColor = 0
# 	top = 0
# 	bottom = 0
# 	left = 0
# 	right = 0

# 	#Finding the edges where the fundus image starts
# 	for col in range(0, cols):
# 		if not np.all(img[:,col, channel] == borderColor):
# 			left = col - 2
# 			break

# 	for col in range(0, cols):
# 		if not np.all(img[:,-col, channel] == borderColor):
# 			right = cols - (col - 2)
# 			break

# 	for row in range(0, rows):
# 		if not np.all(img[row,:, channel] == borderColor):
# 			top = row - 2
# 			break

# 	for row in range(0, rows):
# 		if not np.all(img[-row,:, channel] == borderColor):
# 			bottom = rows - (row - 2)
# 			break

# 	#In case there is no extra edge
# 	if left < 0:
# 		left = 0
# 	if right > cols:
# 		right = cols
# 	if top < 0:
# 		top = 0
# 	if bottom > rows:
# 		bottom = rows

# 	img = img[top:bottom, left:right]

# 	return img

def remove_optic_disc(img):
	'''Detects optic disk centre and then provides thesholding to remove it'''

	# Extracting the red channel of the image
	imgR = img[:,:,2]
	kernel = np.ones((3,3))

	#Dialating and eroding the image to remove the blood vessels
	grayDil = cv2.dilate(imgR, kernel, iterations = 7)
	grayEr = cv2.erode(grayDil, kernel, iterations = 3)

	#Reducing salt and pepper noise introduced by the dilation and erosion
	gray = cv2.medianBlur(np.float32(grayEr), 5)

	#Calculating the edges through the Sobel method
	grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)

	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
	
	#Defining the kernel to find smaller circles
	kernel = np.array([[0,1,0],
					   [1,0,1],
					   [0,1,0]])
	imgF = cv2.filter2D(grad, 0, kernel)

	#Darkening the smaller circles and the image
	imgC = cv2.filter2D(imgF, 0, kernel/100)

	#Defining the kernel to find larger circles
	kernel = np.array([[0, 0, 1, 0, 0],
					   [0, 1, 0, 1, 0],
					   [1, 0, 0, 0, 1],
					   [0, 1, 0, 0, 0],
					   [0, 0, 1, 0, 0]])

	#Darkening the larger circles and the image
	imgC = cv2.filter2D(imgF, 0, kernel/25)

	#Removing salt and pepper noise introduced due to the filtering
	imgCB = cv2.bilateralFilter(imgC, 15, 75, 75)
	imgCB = cv2.medianBlur(imgCB, 7)

	#Creating a binary image for edges
	_, thresh = cv2.threshold(imgCB, 13, 255, cv2.THRESH_BINARY)

	#Find contours. If the length of the contours is not a fixed range, erode the image till atleast one contour in the rang eis detected
	#A maximum of 10 erosions will be performed
	erosion = 6
	while True:
		candidates = 0
		threshE = cv2.erode(thresh, np.ones((2,2)), iterations = erosion)
		threshD = cv2.dilate(threshE, np.ones((4,4)), iterations = 4)
		contours = cv2.findContours(threshD, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		for c in contours:
			if c.shape[0] > 100 and c.shape[0] < 500:
				candidates += 1
		if candidates == 0 and erosion < 10:
			erosion += 1
		else:
			break

	#Draw the contours on the original image
	for c in contours:
		if c.shape[0] > 100 and c.shape[0] < 500:
				# compute the center of the contour
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			# draw the contour and center of the shape on the image
			cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
			cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)

	return img

def enhance(img):
	'''Enhances the image'''
	#Enhancing the contrast
	clahe = cv2.createCLAHE(clipLimit = 3)
	imgNormR = clahe.apply(img[:,:,0])
	imgNormG = clahe.apply(img[:,:,1])
	imgNormB = clahe.apply(img[:,:,2])

	imgNorm = np.stack((imgNormR, imgNormG, imgNormB), axis = -1)
	return imgNorm

def preprocess(image):
	'''Calls all the pre-processing steps'''
	#In case the file passed is a .DS_Store
	if 'DS' in image:
		return 0

	#Read image into a numpy array
	try:
		img = cv2.imread(image, -1)
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
			# cv2.imwrite('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/preprocessed/' + image, img)
			img = cv2.imread(image)

		else:
			print(f'{image} is None')
	except:
		print(f'Writing error in image {image}')

	return 1

	#Display image. Only for testing
	# cv2.imshow(image,img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

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
			new_image = enhance(img)

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

def divide_dataset(dist):

	ntraining = [20648, 1955, 4234, 699, 567]
	nValidation = [5161, 488, 1058, 174, 141]

	training = {0: [], 1: [], 2: [], 3: [], 4: []}
	validation = {0: [], 1: [], 2: [], 3: [], 4: []}

	for label in dist:
		random.shuffle(dist[label])
		training[label] = dist[label][0:ntraining[label]]
		validation[label] = dist[label][-nValidation[label]:]

	return training, validation

def main():
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
		
		imageDistribution = get_images_distribution()

		#Dividing dataset into training and validation set
		trainingData, validationData = divide_dataset(imageDistribution)
		
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
		with open('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/trainingLabels.csv','w') as training:
			trainingWriter = csv.writer(training)
			trainingWriter.writerow(['img-code', 'class'])
			for label in trainingData:
				for image in trainingData[label]:
					trainingWriter.writerow([image, label])

		#Saving the labels in training.csv
		with open('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/validationLabels.csv','w') as validation:
			validationWriter = csv.writer(validation)
			validationWriter.writerow(['img-code', 'class'])
			for label in validationData:
				for image in validationData[label]:
					validationWriter.writerow([image, label])

		# #Moving the files into separate test and validation folders
		for label in validationData:
			for image in validationData[label]:
				try:
					os.rename('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train_preprocessed/'+image + '.jpeg', '/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/validation_preprocessed/'+image+'.jpeg')
				except:
					print(f'Issue in {image}')

		#Post this, rename the folder named 'preprocessing' to 'validation'
		
if __name__ == '__main__':
	main()