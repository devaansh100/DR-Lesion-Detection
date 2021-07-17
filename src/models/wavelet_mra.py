import matplotlib.pyplot as plt
import numpy as np
import os
import pywt
import sys
import torchvision.transforms as transforms
sys.path.append('../config/')

from config import read_config
from dataset import DRDataset
from normalize import find_statistic_param
from saliency_map import create_saliency_map

config = read_config()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 10})

validation_original   = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(800)]))
# mean_val, std_val = find_statistic_param(validation_original)
validation_normalised = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(800), transforms.Normalize(mean=[0.501910388469696, 0.5028203129768372, 0.5062801241874695], std=[0.042280834168195724, 0.036873284727334976, 0.026622962206602097])]))

img, img_preprocessed, grayscale_cam, img_saliency_map = create_saliency_map(validation_original, validation_normalised, 1211)

A = img_saliency_map
B = np.mean(A, -1)

n = 2
w = 'db1'
coeffs = pywt.wavedec2(B, wavelet = w, level = n)

coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
	coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level]]

arr, coeff_slices = pywt.coeff_to_array(coeffs)	

plt.imshow(arr, cmap = 'gray_r', vmin = -0.25, vmax = 0.75)
plt.rcParams['figure.figsize'] = [16,16]
fig = plt.figure(figsize(18,16))
plt.show()