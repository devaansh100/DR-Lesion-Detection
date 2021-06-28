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
		label = entries[i][1]
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
			cv2.imwrite('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/preprocessed/' + image, img)
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

	for i in range(nAugmentations):
		try:
			img = cv2.imread(image)
		except:
			continue

		option = random.randint(1,4)
		new_image = img
		#Randomly choose to flip horizontally, vertically, or both
		if option == 1:
			flipCode = random.randint(-1, 1)
			new_image = cv2.flip(img, flipCode)

		#Rotate the image by a random angle
		elif option == 2:
			angle = random.random()*360
			row,col = img.shape
			center = tuple(np.array([row,col])/2)
			rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
			new_image = cv2.warpAffine(image, rot_mat, (col,row))

		#Enhance the image by the CLAHE operator 
		elif option == 3:
			new_image = enhance(img)

		#Add Gaussian noise
		elif option == 4:
			new_image = skimage.util.random_noise(img, mode = 'gaussian', seed = None, clip = True, **kwargs)

		try:
			#Saving the augmented image
			cv2.imwrite('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/preprocessed/' + image + '-'+str(i+1), new_image)
		except:
			continue

def divide_dataset(dist):

	ntraining = [20648, 3909, 4234, 3965, 4190]
	nValidation = [5162, 977, 1058,  991, 1048]

	training = {0: [], 1: [], 2: [], 3: [], 4: []}
	validation = {0: [], 1: [], 2: [], 3: [], 4: []}

	for label in dist:
		dist[label] = random.shuffle(dist[label])
		training[label] = dist[label][0:ntraining[label]]
		validation[label] = dist[label][-nValidation[label]:]

	return training, validation

def main():
	'''Creates a thread for each image'''
	with concurrent.futures.ThreadPoolExecutor() as executor:
		
		# Pre-processing all training images
		# images = get_images()
		# results = list(tqdm(executor.map(preprocess, images), total=len(images)))

		#Augmenting all pre-processed images
		imageDistribution = get_images_distribution()
		nAugmentations = [0, 1, 0, 6, 5]
		os.chdir('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/preprocessed/')
		augmented1 = list(tqdm(executor.map(augment, imageDistribution[1], nAugmentations[1]), total=len(imageDistribution[1])))
		augmented3 = list(tqdm(executor.map(augment, imageDistribution[3], nAugmentations[3]), total=len(imageDistribution[3]))) 
		augmented4 = list(tqdm(executor.map(augment, imageDistribution[4], nAugmentations[4]), total=len(imageDistribution[4]))) 

		#Adding the new images into the distribution
		for label in imageDistribution:
			for image in imageDistribution[label]:
				for i in range(0, nAugmentations[label]):
					imageDistribution[label].append(image + '-' + str(i+1))

		'''
		Label distribution  - [25810, 2443, 5292, 708, 873]
		Distruibution of nAugmentations - [0, 1, 0, 6, 5]
		Total data distribution - [25810, 4886, 5292, 4956, 5238]
		Distribution in training - [20648, 3909, 4234, 3965, 4190]
		Distribution in validation - [5162, 977, 1058,  991, 1048]
		'''
		#Dividing dataset into training and validation set
		trainingData, validationData = divide_dataset(imageDistribution)

		#Saving the labels in training.csv
		with training open('trainingLabels.csv',w):
			training.writerow(['img-code', 'class'])
			for label in training:
				for image in training[label]:
					training.writerow([image, label])

		#Saving the labels in training.csv
		with validation open('validationLabels.csv',w):
			validation.writerow(['img-code', 'class'])
			for label in validation:
				for image in validation[label]:
					validation.writerow([image, label])

		#Moving the files to separate test and validation folders


	# Step 1: Categorise the images as good and poor and remove the poor images(should we do this? I mean the model is based on the fact that we want to work with poor quality images as well). If yes, how?
	# Step 2: Find the distribution again
	# Step 3: Get 15000 images which don't have DR, 15000 images which do have DR - augment each grade accordingly. If the number required is too high, add more augmentation methods
	# Step 4: Finalise the size and the number of channels that would be required, modify each code accordingly
	# Step 5: Ensure that the division into training and validation is done correctly
	# Step 6: I was going to use Resnet but not many people have really used that one. Going with the efficient net model - need 3 color channels and a specific size. Aso, will probably have to retrain that model since pre-trained isnt available. Will check if I should use weights from imagenet or not
	# Step 7: The pre-processing will have to be done as per this, also adding gaussian noise to all colour channels was giving good sitnct features.
		
if __name__ == '__main__':
	main()