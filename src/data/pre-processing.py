import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_images():
	'''Gets image names from the images directory'''

	os.chdir('images')
	images = os.listdir()[1:]
	return images

def mask(image):
	'''Creates a mask for the image to remove outer black borders'''

	#Creating the mask. Remove black pixels and change to white pixels
	img = cv2.imread(image, -1)[:,:,2]
	maskedImg = np.where(img < 30, 255, img)
	return maskedImg

def main():
	images = get_images()
	for image in images:
		img = mask(image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	

    # img = remove_optic_disc(img)
    # img = normalise(img)
    # img = accentuate(img)	

if __name__ == '__main__':
	main()