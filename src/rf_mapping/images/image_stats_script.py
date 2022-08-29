"""
Code to summarize the natural images used in the research.

Why this script?
There are a lot of natural images that consistently rank the top-N and bottom-
N images. We would like to make sure this is not due to those images having
exceptional pixel values.

Tony Fu, August 28, 2022
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
# from torchvision.models import VGG16_Weights
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.image import preprocess_img_to_tensor
from src.rf_mapping.hook import HookFunctionBase, ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
import src.rf_mapping.constants as c
from src.rf_mapping.net import get_truncated_model

# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
num_images = 50000
batch_size = 100
top_n = 100
yn, xn = (227, 227)
this_is_a_test_run = False

# Please double-check the directories:y
img_dir = c.IMG_DIR
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'images', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'images')

###############################################################################

img_means = np.zeros((num_images,2))
img_stds = np.zeros((num_images,2))
img_maxs = np.zeros((num_images,2))
img_mins = np.zeros((num_images,2))

image_summary_txt_path = os.path.join(result_dir, f"image_summary.txt")
if os.path.exists(image_summary_txt_path):
    os.remove(image_summary_txt_path)

for img_i in tqdm(range(num_images)):
    if this_is_a_test_run and img_i > 1000:
        break

    img_path = os.path.join(img_dir, f"{img_i}.npy")
    img = np.load(img_path)
    img_means[img_i,0] = img.mean()
    img_stds[img_i,0] = img.std()
    img_maxs[img_i,0] = img.max()
    img_mins[img_i,0] = img.min()

    img_tensor = preprocess_img_to_tensor(img)
    img_means[img_i,1] = img_tensor.mean()
    img_stds[img_i,1] = img_tensor.std()
    img_maxs[img_i,1] = img_tensor.max()
    img_mins[img_i,1] = img_tensor.min()

    
    with open(image_summary_txt_path, 'a') as f:
        f.write(f"{img_i} ")
        f.write(f"{img_means[img_i, 0]:.4f} {img_means[img_i, 1]:.4f} ")
        f.write(f"{img_stds[img_i, 0]:.4f} {img_stds[img_i, 1]:.4f} ")
        f.write(f"{img_maxs[img_i, 0]:.4f} {img_maxs[img_i, 1]:.4f} ")
        f.write(f"{img_mins[img_i, 0]:.4f} {img_mins[img_i, 1]:.4f}\n")
    

image_summary_pdf_path = os.path.join(result_dir, f"image_summary.pdf")
fontsize = 14
num_bins = 40
with PdfPages(image_summary_pdf_path) as pdf:
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Statistics of 50,000 ImageNet Images: Before vs. After Normalization", fontsize=26)
    
    plt.subplot(2,4,1)
    plt.hist(img_means[:,0], bins=num_bins)
    plt.xlabel("mean", fontsize=fontsize)
    plt.ylabel("Before")
    
    plt.subplot(2,4,2)
    plt.hist(img_stds[:,0], bins=num_bins)
    plt.xlabel("std", fontsize=fontsize)
    
    plt.subplot(2,4,3)
    plt.hist(img_maxs[:,0], bins=num_bins)
    plt.xlabel("max", fontsize=fontsize)
    
    plt.subplot(2,4,4)
    plt.hist(img_mins[:,0], bins=num_bins)
    plt.xlabel("min", fontsize=fontsize)
    
    plt.subplot(2,4,5)
    plt.hist(img_means[:,1], bins=num_bins)
    plt.xlabel("mean", fontsize=fontsize)
    plt.ylabel("After")
    
    plt.subplot(2,4,6)
    plt.hist(img_stds[:,1], bins=num_bins)
    plt.xlabel("std", fontsize=fontsize)
    
    plt.subplot(2,4,7)
    plt.hist(img_maxs[:,1], bins=num_bins)
    plt.xlabel("max", fontsize=fontsize)
    
    plt.subplot(2,4,8)
    plt.hist(img_mins[:,1], bins=num_bins)
    plt.xlabel("min", fontsize=fontsize)
    
    pdf.savefig()
    plt.close()
