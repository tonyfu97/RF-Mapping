"""
Performs a gradient method on the top/bottom N image patches, then save their
sums.

Tony Fu, June 29, 2022
"""

import os
import sys

import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt

sys.path.append('../../..')
from src.rf_mapping.spatial import get_rf_sizes, SpatialIndexConverter
from src.rf_mapping.image import one_sided_zero_pad, preprocess_img_for_plot
from src.rf_mapping.guided_backprop import GuidedBackprop
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.reproducibility import set_seeds
import src.rf_mapping.constants as c

# Please specify some details here:
set_seeds()
model = models.alexnet(pretrained=False).to(c.DEVICE)
model_name = "alexnet"
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
sum_modes = ['abs']
top_n = 100
this_is_a_test_run = True
is_random = True

# Please double-check the directories:
img_dir = c.IMG_DIR

if is_random:
    index_dir = os.path.join(c.RESULTS_DIR, 'ground_truth',
                             'top_n_random', model_name)
    result_dir = os.path.join(c.RESULTS_DIR, 'ground_truth',
                              'backprop_sum_random', model_name)
else:
    index_dir = os.path.join(c.RESULTS_DIR, 'ground_truth',
                             'top_n', model_name)
    result_dir = os.path.join(c.RESULTS_DIR, 'ground_truth',
                              'backprop_sum', model_name)

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Delete previous results from result directories.
for sum_mode in sum_modes:
    if this_is_a_test_run:
        result_dir_with_sum_mode = os.path.join(result_dir, 'test')
    else:
        result_dir_with_sum_mode = os.path.join(result_dir, sum_mode)
    delete_all_npy_files(result_dir_with_sum_mode)

# Initiate helper objects.
converter = SpatialIndexConverter(model, (227, 227))

# Get info of conv layers.
layer_indices, rf_sizes = get_rf_sizes(model, (227, 227), nn.Conv2d)


def add_patch_to_sum(new_patch, sum, sum_mode):
    """
    Add a new patch to the sum patch.
    
    sum_mode determines what operation is performed on the new patch's pixels
    before adding it to the cummulative sum.
    """
    if sum_mode == 'sum':
        sum += new_patch/new_patch.max()
    elif sum_mode == 'abs':
        new_patch = np.absolute(new_patch)
        sum += new_patch/new_patch.max()
    elif sum_mode == 'sqr':
        new_patch = np.square(new_patch)
        sum += new_patch/new_patch.max()
    elif sum_mode == 'relu':
        new_patch[new_patch<0] = 0
        sum += new_patch/new_patch.max()
    else:
        raise KeyError("sum mode must be 'abs', 'sqr', or 'relu'.")
    return sum


def get_grad_patch(img, layer_idx, unit_i, spatial_idx, rf_size):
    """
    Returns
    -------
    gradient_patch_padded : numpy.array
        Gradient patches generated with the grad_method. Each patch will have
        the dimension: {rf_size} x {rf_size}. The patch will be padded with
        zeros if necessary.
    """
    vx_min, hx_min, vx_max, hx_max = converter.convert(spatial_idx, layer_idx, 0,
                                                       is_forward=False)
    grad_map = grad_method.generate_gradients(img, unit_i, spatial_idx)
    grad_patch = grad_map[:, vx_min:vx_max+1, hx_min:hx_max+1]
    
    # Average over the color channels.
    grad_patch = np.mean(grad_patch, axis=0)
    
    grad_patch_padded = one_sided_zero_pad(grad_patch, rf_size,
                                           (vx_min, hx_min, vx_max, hx_max))
    return grad_patch_padded


# Loop through layers...
for conv_i, rf_size in enumerate(rf_sizes):
    layer_idx = layer_indices[conv_i]
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indices = np.load(index_path).astype(int)
    num_units, num_images, _ = max_min_indices.shape
    grad_method = GuidedBackprop(model, layer_idx)
    print(f"Summing gradient results for {layer_name}...")

    # Initializing arrays:
    max_sum = np.zeros((len(sum_modes), num_units, rf_size[0], rf_size[1]))
    min_sum = np.zeros((len(sum_modes), num_units, rf_size[0], rf_size[1]))

    for unit_i in tqdm(range(num_units)):
        # Do only the first 5 unit during testing phase
        if this_is_a_test_run and unit_i >= 5:
            break

        for img_i in range(top_n):
            try:
                # Fatch indices
                max_img_idx, max_idx, min_img_idx, min_idx = max_min_indices[unit_i, img_i, :]
            except:
                print(f"top_n of {top_n} exceeds the number of images in the ranking data.")
                break

            # Top N images:
            max_img_path = os.path.join(img_dir, f"{max_img_idx}.npy")
            max_img = np.load(max_img_path)
            max_grad_patch_padded = get_grad_patch(max_img, layer_idx, unit_i, max_idx, rf_size)
            for i, sum_mode in enumerate(sum_modes):
                max_sum[i, unit_i, ...] = add_patch_to_sum(max_grad_patch_padded, max_sum[i, unit_i, ...], sum_mode)
            
            print(np.sum(max_sum < 0))

            # Bottom N images:
            min_img_path = os.path.join(img_dir, f"{min_img_idx}.npy")
            min_img = np.load(min_img_path)
            min_grad_patch_padded = get_grad_patch(min_img, layer_idx, unit_i, min_idx, rf_size)
            for i, sum_mode in enumerate(sum_modes):
                min_sum[i, unit_i, ...] = add_patch_to_sum(min_grad_patch_padded, min_sum[i, unit_i, ...], sum_mode)

        if this_is_a_test_run:
            plt.figure(figsize=(15,5))
            sum_mode_idx = 0
            plt.suptitle(f"Gradient average of image patches ({layer_name} no.{unit_i}, "
                        f"sum mode: {sum_modes[sum_mode_idx]})", fontsize=20)

            plt.subplot(1, 3, 1)
            plt.imshow(preprocess_img_for_plot(max_sum[sum_mode_idx, unit_i, ...]), cmap='gray')
            plt.title("max", fontsize=16)
            plt.subplot(1, 3, 2)
            plt.imshow(preprocess_img_for_plot(min_sum[sum_mode_idx, unit_i, ...]), cmap='gray')
            plt.title("min", fontsize=16)
            plt.subplot(1, 3, 3)
            both_sum = max_sum[sum_mode_idx, unit_i, ...] + min_sum[sum_mode_idx, unit_i, ...]
            plt.imshow(preprocess_img_for_plot(both_sum), cmap='gray')
            plt.title("max + min", fontsize=16)
            plt.show()

    # Normalize
    max_sum_norm = max_sum/num_units
    min_sum_norm = min_sum/num_units
    both_sum_norm = (max_sum_norm + min_sum_norm)/2

    # Save results.
    for i, sum_mode in enumerate(sum_modes):
        if this_is_a_test_run:
            sum_mode = 'test'
        max_result_path = os.path.join(result_dir, sum_mode, f"conv{conv_i+1}_max.npy")
        min_result_path = os.path.join(result_dir, sum_mode, f"conv{conv_i+1}_min.npy")
        np.save(max_result_path, max_sum_norm[i,...])
        np.save(min_result_path, min_sum_norm[i,...])
