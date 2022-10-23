"""
Script to plot the example units for different levels of direct correlation.

Tony Fu, October 6th, 2022
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
# model = models.alexnet(pretrained=True).to(c.DEVICE)
# model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = 'resnet18'
# image_shape = (227, 227)
this_is_a_test_run = False

# font_size = 20
# r_val_threshold = 0.7
# to_plot_pdf = False

model_name = 'alexnet'
num_layers = 5

sigma_rf_ratio = 1/30
max_or_min = 'max'
map_name = 'rfmp4a'

unit_w_low_r = {'layer_name': 'conv3', 'unit_i': 50}
unit_w_mid_r   = {'layer_name': 'conv4', 'unit_i': 7}
unit_w_high_r  = {'layer_name': 'conv2', 'unit_i': 2}

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

apply_fix = False
fix_pix_thres = 0.1

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
    elif map_name == 'block':
        mapping_path = os.path.join(mapping_dir,
                                    'block',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_blockmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.transpose(maps, (0,2,3,1))  # Need the color channel for plots.
    else:
        raise KeyError(f"{map_name} does not exist.")

def smooth_and_normalize_map(map, sigma_rf_ratio):
    sigma = sigma_rf_ratio * map.shape[0]
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
    
def plot_r_val(r_val, font_size):
    plt.xticks([])
    plt.yticks([])
    plt.text(0.1, 0.7, f"r = {r_val:.4f}", fontsize=font_size, color='w')
    if math.isfinite(r_val):
        ax = plt.gca()
        ax.set_facecolor((r_val/2 + 0.5, 1 - abs(r_val), 1-r_val/2 - 0.5))


#########################  LOAD MAP CORRELATIONS  #############################

max_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                model_name, f"max_map_r_0.0333.txt")

max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)

# map_corr_dict[model_name] =\
#         max_map_corr_df[['LAYER', 'UNIT', 'gt_composite_vs_occlude_composite']].copy()
        
################################ MAKE PDF #####################################

fix_or_not = '_fix' if apply_fix else ''
pdf_path = os.path.join(result_dir, f"{model_name}_{map_name}_correlation_example{fix_or_not}.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(18, 15))
    
    for plot_i, unit_data in enumerate([unit_w_low_r, unit_w_mid_r, unit_w_high_r]):
        gt_map = load_maps('gt', unit_data['layer_name'], max_or_min)[unit_data['unit_i']]
        gt_map = smooth_and_normalize_map(gt_map, sigma_rf_ratio)
        rfmp_map = load_maps(map_name, unit_data['layer_name'], max_or_min)[unit_data['unit_i']]
        rfmp_map = smooth_and_normalize_map(rfmp_map, sigma_rf_ratio)
        
        if len(rfmp_map.shape) == 3:  # if there is a color channel
            rfmp_map_no_color = np.mean(rfmp_map, axis=2)
        else:
            rfmp_map_no_color = rfmp_map

        if apply_fix:
            gt_map_flatten = gt_map.flatten()
            rfmp_map_flatten = rfmp_map_no_color.flatten()
            idx_to_keep = (gt_map_flatten > fix_pix_thres) & (rfmp_map_flatten > fix_pix_thres)
            gt_map_flatten = gt_map_flatten[idx_to_keep]
            rfmp_map_flatten = rfmp_map_flatten[idx_to_keep]
        else:
            gt_map_flatten = gt_map.flatten()
            rfmp_map_flatten = rfmp_map_no_color.flatten()
        r_val, _ = pearsonr(gt_map_flatten, rfmp_map_flatten)
        
        plt.subplot(3, 3, plot_i + 1)
        plt.imshow(gt_map, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if plot_i == 0:
            plt.ylabel('Ground truth', fontsize=18)
        plt.title(f"{unit_data['layer_name']}-{unit_data['unit_i']} (r = {r_val:.4f})", fontsize=18)
        
        plt.subplot(3, 3, plot_i + 4)
        plt.imshow(rfmp_map, cmap='gray')
        if plot_i == 0:
            plt.ylabel(f"{map_name}", fontsize=18)
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(3, 3, plot_i + 7)
        plt.scatter(gt_map_flatten, rfmp_map_flatten, alpha=0.05)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if plot_i == 0:
            plt.xlabel('GT normalized pix value', fontsize=14)
            plt.ylabel(f'{map_name} normalized pix value', fontsize=14)
        plt.gca().set_aspect('equal')
    
    pdf.savefig()
    plt.show()
    plt.close()
    
    
    plt.figure(figsize=(10, 7))
    corr_name = f"gt_vs_{map_name}"
    corr_means = max_map_corr_df.groupby('LAYER').mean()[corr_name]
    corr_stds = max_map_corr_df.groupby('LAYER').std()[corr_name]
    plt.errorbar(np.arange(5), corr_means, yerr=corr_stds)
    plt.ylim([-0.1, 1.1])
    plt.title(f"{corr_name}", fontsize=18)
    plt.ylabel('correlation', fontsize=18)
    
    pdf.savefig()
    plt.show()
    plt.close()
