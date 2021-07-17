from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from dataset import DRDataset
from transfer_learning import Pretrained
from normalize import find_statistic_param

import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import torch
import sys
sys.path.append('../config/')
from config import read_config

def create_saliency_map(dataset_original, dataset_normalised, index, plot = False):
	config = read_config()

	googlenet = models.googlenet(pretrained=True)
	for param in googlenet.parameters():
		param.requires_grad = False
	model = Pretrained(googlenet)
	model.load_state_dict(torch.load(config['MODEL_PATH'] + 'final_googlenet.pth', map_location = config['DEVICE']))

	target_layer = list(model.modules())[-10]
	img, truth = dataset_normalised.__getitem__(index)
	img_preprocessed, _ = dataset_original.__getitem__(index)
	img = img.view(1, 3, 800, 800)
	input_tensor = img
	cam = GradCAMPlusPlus(model=model, target_layer=target_layer)
	target_category = None
	grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
	grayscale_cam = grayscale_cam[0, :]
	img, _ = inf_validation.__getitem__(index)
	img = img.permute(1,2,0)
	img = img.numpy()
	img_saliency_map = cv2.addWeighted(img[:,:,1], 0.5, grayscale_cam, 0.5, 0)
	
	img_preprocessed = img_preprocessed.permute(1,2,0)
	img_preprocessed = img_preprocessed.numpy()

	if plot:
		f, axarr = plt.subplots(2,2, figsize = (12,12))
		axarr[0,0].set_title('saliency map + preprocessed + normalised')
		axarr[0,0].imshow(img_saliency_map)
		axarr[0,1].set_title('preprocessed + normalised')
		axarr[0,1].imshow(img)
		axarr[1,0].set_title('saliency map')
		axarr[1,0].imshow(grayscale_cam)
		axarr[1,1].set_title('preprocessed')
		axarr[1,1].imshow(img_preprocessed)

	return (img, img_preprocessed, grayscale_cam, img_saliency_map)