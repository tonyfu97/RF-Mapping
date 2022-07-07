"""
Make a pdf of the top N natural images and the gradient maps.

Tony Fu, June 26, 2022
"""
import os
import sys

import numpy as np
from pathlib import Path
import torch
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('..')
from hook import ConvUnitCounter
from image import preprocess_img_for_plot, make_box
from spatial import SpatialIndexConverter
from guided_backprop import GuidedBackprop
import constants as c

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = "alexnet"
top_n = 5
grad_method = GuidedBackprop(model)

# Please double-check the directories:
img_dir = c.IMG_DIR
index_dir = c.REPO_DIR + f'/results/ground_truth/top_n/{model_name}'
result_dir = index_dir

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take hours to run. Are you sure? ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Initiate helper objects.
converter = SpatialIndexConverter(model, (227, 227))
conv_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, nums_units = conv_counter.count()


def plot_one_img(img_idx, box):
    """Plots the image and draw the red box."""
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    img = preprocess_img_for_plot(img)
    plt.imshow(img)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax = plt.gca()
    ax.add_patch(make_box(box))


def plot_one_grad_map(img_idx, layer_idx, unit_idx, patch_idx, box):
    """Plots the target unit's gradient map for the image."""
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp_map = grad_method.generate_gradients(img, layer_idx, unit_idx, patch_idx)
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax = plt.gca()
    ax.add_patch(make_box(box))


for conv_i, layer_idx in enumerate(layer_indices):
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indices = np.load(index_path).astype(int)  
    # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]

    num_units = nums_units[conv_i]
    print(f"Making pdf for {layer_name}...")
    
    pdf_path = os.path.join(result_dir, f"{layer_name}.pdf")
    with PdfPages(pdf_path) as pdf:
        
        for unit_i in tqdm(range(num_units)):
            # Fatch indices
            max_n_img_indices = max_min_indices[unit_i, :top_n, 0]
            max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]
            min_n_img_indices = max_min_indices[unit_i, :top_n, 2]
            min_n_patch_indices = max_min_indices[unit_i, :top_n, 3]

            plt.figure(figsize=(20, 15))
            plt.suptitle(f"{layer_name} unit no.{unit_i}", fontsize=20)

            # Top N images and gradient patches:
            for i, (max_img_idx, max_patch_idx) in enumerate(zip(max_n_img_indices,
                                                            max_n_patch_indices)):
                box = converter.convert(max_patch_idx, layer_idx, 0, is_forward=False)
                plt.subplot(4, top_n, i+1)
                plot_one_img(max_img_idx, box)
                plt.title(f"top {i+1} image")
                
                plt.subplot(4, top_n, i+top_n+1)
                plot_one_grad_map(max_img_idx, layer_idx, unit_i, max_patch_idx, box)
                plt.title(f"top {i+1} gradient")

            # Bottom N images and gradient patches:
            for i, (min_img_idx, min_patch_idx) in enumerate(zip(min_n_img_indices,
                                                            min_n_patch_indices)):
                box = converter.convert(min_patch_idx, layer_idx, 0, is_forward=False)
                plt.subplot(4, top_n, i+2*top_n+1)
                plot_one_img(min_img_idx, box)
                plt.title(f"bottom {i+1} image")
                
                plt.subplot(4, top_n, i+3*top_n+1)
                plot_one_grad_map(min_img_idx, layer_idx, unit_i, min_patch_idx, box)
                plt.title(f"bottom {i+1} gradient")
            
            pdf.savefig()
            plt.close()
