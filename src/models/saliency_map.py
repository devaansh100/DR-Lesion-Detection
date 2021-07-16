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

config = read_config()

googlenet = models.googlenet(pretrained=True)
for param in googlenet.parameters():
  param.requires_grad = False
model = Pretrained(googlenet)

i = 2379
validation_original   = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(800)]))
mean_val, std_val = find_statistic_param(validation_original)
validation_normalised = DRDataset(config['VAL_LABELS'], config['VAL_IMG'], transforms.Compose([transforms.ToTensor(), transforms.Resize(800), transforms.Normalize(mean=mean_val, std = std_val)]))
target_layer = list(model.modules())[-10]
img, truth = validation_normalised.__getitem__(i)
pr, _ = validation_original.__getitem__(i)
img = img.view(1, 3, 800, 800)
input_tensor = img
cam = GradCAMPlusPlus(model=model, target_layer=target_layer)
target_category = None
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
grayscale_cam = grayscale_cam[0, :]
img, _ = inf_validation.__getitem__(i)
img = img.permute(1,2,0)
img = img.numpy()
imgF = cv2.addWeighted(img[:,:,1], 0.5, grayscale_cam, 0.5, 0)
preprocessed = preprocessed.permute(1,2,0)
preprocessed = preprocessed.numpy()

f, axarr = plt.subplots(2,2, figsize = (12,12))
axarr[0,0].set_title('saliency map + preprocessed + normalised')
axarr[0,0].imshow(imgF)
axarr[0,1].set_title('preprocessed + normalised')
axarr[0,1].imshow(img)
axarr[1,0].set_title('saliency map')
axarr[1,0].imshow(grayscale_cam)
axarr[1,1].set_title('preprocessed')
axarr[1,1].imshow(preprocessed)