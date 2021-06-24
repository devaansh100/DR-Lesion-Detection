import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_images():
	'''Gets image names from the images directory'''

	os.chdir('../../data/train/')
	images = os.listdir()[1:] #starting from 1 to avoid the .DS_Store
	return images

def mask(img):
	'''Creates a mask for the image to remove outer black borders'''

	#Creating the mask. Remove black pixels and change to white pixels
	maskedImg = np.where(img < 30, 255, img)

	#Cropping the image to remove excess borders
	rows, cols = maskedImg.shape
	top = 0
	bottom = 0
	left = 0
	right = 0
	for col in range(0, cols):
		if not np.all(maskedImg[:,col] == 255):
			left = col - 5
			break

	for col in range(0, cols):
		if not np.all(maskedImg[:,-col] == 255):
			right = cols - (col - 5)
			break

	for row in range(0, rows):
		if not np.all(maskedImg[row,:] == 255):
			top = row - 5
			break

	for row in range(0, rows):
		if not np.all(maskedImg[-row,:] == 255):
			bottom = rows - (row - 5)
			break

	#In case there is no extra white edge
	if left < 0:
		left = 0
	if right > cols:
		right = cols
	if top < 0:
		top = 0
	if bottom < rows:
		bottom = rows

	maskedImg = maskedImg[top:bottom, left:right]

	return maskedImg

# def remove_optic_disc(img):
	# img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)

def main():
	images = get_images()
	for image in images:
		img = cv2.imread(image, -1)[:,:,1]
		cv2.imshow('before', img)
		img = mask(img)
		cv2.imshow('after', img)
		# img = remove_optic_disc(img)
	    # img = normalise(img)
	    # img = accentuate(img)	
		# cv2.imshow(image, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()		

if __name__ == '__main__':
	main()