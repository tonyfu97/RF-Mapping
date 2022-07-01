"""
Performs a gradient method on the top/bottom N image patches, then save their
sums.

Tony Fu, June 29, 2022
"""

import os
import sys

import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

sys.path.append('..')
from hook import get_rf_sizes, SpatialIndexConverter
from image import one_sided_zero_pad, preprocess_img_for_plot
from guided_backprop import GuidedBackprop
from files import delete_all_npy_files

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Please specify some details here:
model = models.alexnet(pretrained = True).to(device)
model_name = "alexnet"
sum_modes = ['abs', 'sqr', 'relu', 'sum']
grad_method = GuidedBackprop(model)
top_n = 1

# Please double-check the directories:
img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"
index_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/ground_truth/top_n/{model_name}')
result_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/ground_truth/backprop_sum/{model_name}')

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input(
        "This code may take hours to run. Are you sure? "\
        f"All .npy files in {result_dir} will be deleted. (y/n): ") 
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Delete previous results from result directories.
for sum_mode in sum_modes:
    result_dir_with_sum_mode = os.path.join(result_dir, sum_mode)
    delete_all_npy_files(result_dir_with_sum_mode)

# Initiate helper objects.
converter = SpatialIndexConverter(model, (227, 227))

# Get info of conv layers.
layer_indicies, rf_sizes = get_rf_sizes(model, (227, 227), nn.Conv2d)


def add_patch_to_sum(sum, new_patch, sum_mode):
    """
    Add a new patch to the sum patch.
    
    sum_mode determines what operation is performed on the new patch's pixels
    before adding it to the cummulative sum.
    """
    if sum_mode == 'sum':
        sum += new_patch
    elif sum_mode == 'abs':
        sum += np.absolute(new_patch)
    elif sum_mode == 'sqr':
        sum += np.square(new_patch)
    elif sum_mode == 'relu':
        new_patch[new_patch<0] = 0
        sum += new_patch
    else:
        raise KeyError("sum mode must be 'abs', 'sqr', or 'relu'.")
    return sum


def get_grad_patch(img, layer_idx, unit_i, spatial_idx, rf_size):
    """
    Returns
    -------
    gradient_patch_padded : numpy.array
        Gradient patches generated with the grad_method. Each patch will have
        the dimension: {rf_size} x {rf_size} x 3. The patch will be padded
        with zeros if necessary.
    """
    vx_min, hx_min, vx_max, hx_max = converter.convert(spatial_idx, layer_idx, 0,
                                                       is_forward=False)
    grad_map = grad_method.generate_gradients(img, layer_idx, unit_i, spatial_idx)
    grad_patch = grad_map[:, vx_min:vx_max+1, hx_min:hx_max+1]
    grad_patch_padded = one_sided_zero_pad(grad_patch, rf_size,
                                           (vx_min, hx_min, vx_max, hx_max))
    return grad_patch_padded


# Loop through layers...
for conv_i, rf_size in enumerate(rf_sizes):
    layer_idx = layer_indicies[conv_i]
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indicies = np.load(index_path).astype(int)
    num_units, num_images, _ = max_min_indicies.shape
    print(f"Summing gradient results for {layer_name}...")
        
    for unit_i in tqdm(range(num_units)):
        max_sum = np.zeros((len(sum_modes), rf_size[0], rf_size[1], 3))
        min_sum = np.zeros((len(sum_modes), rf_size[0], rf_size[1], 3))
        
        for img_i in range(top_n):
            try:
                # Fatch indicies
                max_img_idx, max_idx, min_img_idx, min_idx = max_min_indicies[unit_i, img_i, :]
            except:
                print(f"top_n of {top_n} exceeds the number of images in the ranking data.")
                break
            
            # Top N images:
            max_img_path = os.path.join(img_dir, f"{max_img_idx}.npy")
            max_img = np.load(max_img_path)
            max_grad_patch_padded = get_grad_patch(max_img, layer_idx, unit_i, max_idx, rf_size)
            for i, sum_mode in enumerate(sum_modes):
                max_sum[i,...] = add_patch_to_sum(max_grad_patch_padded, max_sum[i,...], sum_mode)
            
            # Bottom N images:
            min_img_path = os.path.join(img_dir, f"{min_img_idx}.npy")
            min_img = np.load(min_img_path)
            min_grad_patch_padded = get_grad_patch(min_img, layer_idx, unit_i, min_idx, rf_size)
            for i, sum_mode in enumerate(sum_modes):
                min_sum[i,...] = add_patch_to_sum(min_grad_patch_padded, min_sum[i,...], sum_mode)

        # Normalize
        # TODO: fix this so that there is no infinity
        max_sum_norm = max_sum/num_units
        min_sum_norm = min_sum/num_units
        both_sum_norm = (max_sum_norm + min_sum_norm)/2

        plt.figure(figsize=(15,5))
        sum_mode_idx = 0
        plt.suptitle(f"conv{conv_i+1} unit no.{unit_i} (sum mode: {sum_mode}", fontsize=24)
        plt.subplot(1, 3, 1)
        plt.imshow(preprocess_img_for_plot(max_sum_norm[sum_mode_idx,...]))
        plt.title("max", fontsize=20)
        plt.subplot(1, 3, 2)
        plt.imshow(preprocess_img_for_plot(min_sum_norm[sum_mode_idx,...]))
        plt.title("min", fontsize=20)
        plt.subplot(1, 3, 3)
        plt.imshow(preprocess_img_for_plot(both_sum_norm[sum_mode_idx,...]))
        plt.title("max + min", fontsize=20)
        plt.show()

        # Save results.
        for i, sum_mode in enumerate(sum_modes):
            max_result_path = os.path.join(result_dir, sum_mode, f"max_conv{conv_i+1}.{unit_i}.npy")
            min_result_path = os.path.join(result_dir, sum_mode, f"min_conv{conv_i+1}.{unit_i}.npy")
            both_result_path = os.path.join(result_dir, sum_mode, f"both_conv{conv_i+1}.{unit_i}.npy")
            np.save(max_result_path, max_sum_norm[i,...])
            np.save(min_result_path, min_sum_norm[i,...])
            np.save(both_result_path, both_sum_norm[i,...])
