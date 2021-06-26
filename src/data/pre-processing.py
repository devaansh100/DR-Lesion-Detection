import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import concurrent.futures
from imutils import contours
from skimage import measure
import imutils


def get_images():
	'''Gets image names from the images directory'''

	os.chdir('../../data/train/')
	images = os.listdir() #starting from 1 to avoid the .DS_Store
	return images

def mask(img):
	'''Creates a mask for the image to remove outer black borders'''

	#Creating the mask. Remove black pixels and change to white pixels
	maskedImg = np.where(maskedImg < 30, 255, maskedImg)

	#Resizing the image
	rows, _, _ = img.shape
	scale = 600/rows if rows > 600 else rows/600
	maskedImg = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
	return maskedImg

def crop_edges(img):
	rows, cols,_ = img.shape

	#Finding the color of the edges
	borderColor = img[0,0]
	top = 0
	bottom = 0
	left = 0
	right = 0

	#Finding the edges where the fundus image starts
	for col in range(0, cols):
		if not np.all(img[:,col,:] == borderColor):
			left = col - 5
			break

	for col in range(0, cols):
		if not np.all(img[:,-col,:] == borderColor):
			right = cols - (col - 5)
			break

	for row in range(0, rows):
		if not np.all(img[row,:,:] == borderColor):
			top = row - 5
			break

	for row in range(0, rows):
		if not np.all(img[-row,:,:] == borderColor):
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

	#Extracting the red channel of the image
	imgR = img[:,:,2]
	kernel = np.ones((3,3))

	#Dialating and eroding the image to remove the blood vessels
	grayDil = cv2.dilate(imgG, kernel, iterations = 7)
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

def preprocess(image):
	#In case the image passed is a .DS_Store
	if 'DS' in image:
		return 0
	img = cv2.imread(image, -1)

	#Resize the image and make the background white
	img = mask(img)

	#Cropping the extra edges
	img = crop_edges(img)

	# img = normalise(img)
	# img = accentuate(img)	
	# cv2.imshow(image, img)
	cv2.imshow(image,img)
	cv2.waitKey()
	cv2.destroyAllWindows()


def main():
	# with concurrent.futures.ProcessPoolExecutor() as executor:
		# images = get_images()
		# results = executor.map(preprocess, images)
	images = get_images()
	preprocess(images[-1])

		
if __name__ == '__main__':
	main()