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

def get_images():
	'''Gets image names from the images directory'''

	os.chdir('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/train/')
	# os.chdir('/Users/devaanshgupta/Desktop/PS-I/DR-Lesion-Detection/data/train/')
	images = os.listdir() #starting from 1 to avoid the .DS_Store
	return images

def get_images_to_augment():
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

	#Converting from BGR to RGB
	maskedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	#Resizing the image
	maskedImg = cv2.resize(maskedImg, (224,224))

	#Creating the mask. Remove black pixels and change to white pixels
	maskedImg = cv2.addWeighted(maskedImg,4, cv2.GaussianBlur(maskedImg , (0,0) , 30) ,-4 ,128)

	return maskedImg

def crop_edges(img):
	'''Crops the extra edges in the image'''
	rows, cols, _ = img.shape

	#Finding the color of the edges
	borderColor = img[0,0]
	top = 0
	bottom = 0
	left = 0
	right = 0

	#Finding the edges where the fundus image starts
	for col in range(0, cols):
		if not np.all(img[:,col, :] == borderColor):
			left = col - 5
			break

	for col in range(0, cols):
		if not np.all(img[:,-col, :] == borderColor):
			right = cols - (col - 5)
			break

	for row in range(0, rows):
		if not np.all(img[row,:, :] == borderColor):
			top = row - 5
			break

	for row in range(0, rows):
		if not np.all(img[-row,:, :] == borderColor):
			bottom = rows - (row - 5)
			break

	#In case there is no extra edge
	if left < 0:
		left = 0
	if right > cols:
		right = cols
	if top < 0:
		top = 0
	if bottom < rows:
		bottom = rows

	#Cropping the images to remove extra edges
	img = img[top:bottom, left:right, :]

	return img

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

	#Resize the image and make the background white
	img = mask(img)

	#Cropping the extra edges
	img = crop_edges(img)

	#Add condition for image of size zero

	#save image
	try:
		cv2.imwrite('/Volumes/Seagate Backup Plus Drive/DR Kaggle Dataset/train_data_unzip/preprocessed/' + image, img)
		# cv2.imwrite('/Users/devaanshgupta/Desktop/PS-I/DR-Lesion-Detection/data/preprocessed/' + image, img)
	except:
		print(f'Writing error in image {image}')

	return 1

	#Display image. Only for testing
	# cv2.imshow(image,img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

def augment(images, nAugmentations):
	'''Performs data augmentation on each image provided in a list'''

	for image in images:
		for i in range(nAugmentations):
			try:
				img = cv2.imread(image)
				if img.shape != (224, 224): #Size in consideration
					img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
					cv2.imwrite(image, img) #Add the image location in the image code
			except:
				continue

			option = random.randint(1,3)
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

			#Convert to HSV 
			elif option == 3:
				new_image = enhance(img)

			try:
				#Saving the augmented image
				cv2.imwrite(image + '-'+str(i), new_image)
			except:
				continue


def main():
	'''Creates a thread for each image'''
	with concurrent.futures.ThreadPoolExecutor() as executor:
		images = get_images()

		results = list(tqdm(executor.map(preprocess, images), total=len(images)))
		# imagesToAugment = get_images_to_augment()

		#Image augmentation and pre-processing has been separated in order to understand the distribution after removing the poor quality images
		# augmented = executor.map(augment, imagesToAugment, nAugmentations)
	# images = get_images()
	# preprocess(images[-1])

	# Step 1: Categorise the images as good and poor and remove the poor images(should we do this? I mean the model is based on the fact that we want to work with poor quality images as well). If yes, how?
	# Step 2: Find the distribution again
	# Step 3: Get 15000 images which don't have DR, 15000 images which do have DR - augment each grade accordingly. If the number required is too high, add more augmentation methods
	# Step 4: Finalise the size and the number of channels that would be required, modify each code accordingly
	# Step 5: Ensure that the division into training and validation is done correctly
	# Step 6: I was going to use Resnet but not many people have really used that one. Going with the efficient net model - need 3 color channels and a specific size. Aso, will probably have to retrain that model since pre-trained isnt available. Will check if I should use weights from imagenet or not
	# Step 7: The pre-processing will have to be done as per this, also adding gaussian noise to all colour channels was giving good sitnct features.
		
if __name__ == '__main__':
	main()