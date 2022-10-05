"""
Script to correlate the maps and plot the maps of each units.

Tony Fu, August 21st, 2022
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = 'resnet18'
image_shape = (227, 227)
this_is_a_test_run = False
r_val_threshold = 0.7
sigma_rf_ratios = [0, 1/120, 1/60, 1/30, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2]
sigma_rf_ratios_str = ['0', '1/120', '1/60', '1/30', '1/20', '1/10', '1/5', '1/4', '1/3', '1/2']

max_or_min = 'max'
layer_name = "conv5"
unit_i = 3

# Result paths:
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR,
                              'results',
                              'compare',
                              'map_correlations',
                              'test')
else:
    result_dir = os.path.join(c.REPO_DIR,
                             'results',
                             'compare',
                             'map_correlations',
                              model_name)

#############################  HELPER FUNCTIONS  ##############################

# Define helper functions:
def load_maps(map_name, layer_name, max_or_min):
    """Loads the maps of the layer."""
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    
    if map_name == 'gt':
        mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    'backprop_sum',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_{max_or_min}.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'gt_composite':
        max_mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    'backprop_sum',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_max.npy")
        min_mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    'backprop_sum',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_min.npy")
        max_map = np.load(max_mapping_path)  # [unit, yn, xn]
        min_map = np.load(min_mapping_path)  # [unit, yn, xn]
        return max_map + min_map
    elif map_name == 'occlude':
        mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_{max_or_min}.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'occlude_composite':
        max_mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_max.npy")
        min_mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_min.npy")
        max_map = np.load(max_mapping_path)  # [unit, yn, xn]
        min_map = np.load(min_mapping_path)  # [unit, yn, xn]
        return max_map + min_map
    elif map_name == 'rfmp4a':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4a',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'rfmp4c7o':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4c7o',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.transpose(maps, (0,2,3,1))  # Need the color channel for plots.
    elif map_name == 'rfmp_sin1':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp_sin1',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_sinemaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'pasu':
        mapping_path = os.path.join(mapping_dir,
                                    'pasu',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_shapemaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    else:
        raise KeyError(f"{map_name} does not exist.")


def smooth_and_normalize_maps(map, sigma):
    smoothed_map = np.empty(map.shape)
    if len(map.shape) == 2:  # [yn, xn]
        if sigma == 0:
            smoothed_map = map
        else:
            smoothed_map = gaussian_filter(map, sigma=sigma)
        if not math.isclose(smoothed_map.max(), 0, abs_tol=10 ** (-5)):
            smoothed_map = smoothed_map/smoothed_map.max()
    elif len(map.shape) == 3:  # [yn, xn, 3]
        for color_i in range(3):
            if sigma == 0:
                smoothed_map[:,:,color_i] = map[:,:,color_i]
            else:
                smoothed_map[:,:,color_i] = gaussian_filter(map[:,:,color_i], sigma=sigma)
        if not math.isclose(smoothed_map.max(), 0, abs_tol=10 ** (-5)):
            smoothed_map = smoothed_map/smoothed_map.max()
    return smoothed_map

###############################################################################

# Load the maps:
gt_map = load_maps('gt', layer_name, max_or_min)[unit_i]
rfmp4a_map = load_maps('rfmp4a', layer_name, max_or_min)[unit_i]
rfmp4c7o_map = load_maps('rfmp4c7o', layer_name, max_or_min)[unit_i]

pdf_path = os.path.join(result_dir,
                        f"{layer_name}_{unit_i}_{max_or_min}_blur_example.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(len(sigma_rf_ratios) * 5, 10))
    plt.suptitle(f"{layer_name} {layer_name} no.{unit_i}", fontsize=24)
    for plot_i, sigma_rf_ratio in enumerate(sigma_rf_ratios):
        sigma = sigma_rf_ratio * gt_map.shape[-1]
        smoothed_gt_map = smooth_and_normalize_maps(gt_map, sigma)
        smoothed_rfmp4a_map = smooth_and_normalize_maps(rfmp4a_map, sigma)
        r_val, _ = pearsonr(smoothed_gt_map.flatten(), smoothed_rfmp4a_map.flatten())

        plt.subplot(2, len(sigma_rf_ratios), plot_i + 1)
        plt.imshow(smoothed_gt_map, cmap='gray')
        plt.title(f"{sigma_rf_ratios_str[plot_i]} (r = {r_val:.4f})", fontsize=14)
        
        plt.subplot(2, len(sigma_rf_ratios), plot_i + 1 + len(sigma_rf_ratios))
        plt.imshow(smoothed_rfmp4a_map, cmap='gray')
    
    pdf.savefig()
    plt.show()
    plt.close()
