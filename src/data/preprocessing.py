import cv2
import numpy as np
import imutils
from imutils import contours

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

	