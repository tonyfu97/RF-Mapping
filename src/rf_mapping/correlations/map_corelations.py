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
this_is_a_test_run = True
max_or_min = 'max'
font_size = 20
r_val_threshold = 0.7
sigma_rf_ratio = 1/30  # From [0, 1/120, 1/60, 1/30, 1/20, 1/10, 1/5, 1/4, 1/2]
to_plot_pdf = True

# ADDING NEW MAP? MODIFY BELOW:
all_map_names = ['gt', 'gt_composite', 'occlude_composite',
                 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']

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

###############################################################################

# Get some layer info:
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

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

def smooth_and_normalize_maps(maps, sigma):
    num_units = maps.shape[0]
    smoothed_maps = np.empty(maps.shape)
    if len(maps.shape) == 3:  # [unit, yn, xn]
        for unit_i in range(num_units):
            if sigma == 0:
                smoothed_maps[unit_i] = maps[unit_i]
            else:
                smoothed_maps[unit_i] = gaussian_filter(maps[unit_i], sigma=sigma)
            if not math.isclose(smoothed_maps[unit_i].max(), 0, abs_tol=10 ** (-5)):
                smoothed_maps[unit_i] = smoothed_maps[unit_i]/smoothed_maps[unit_i].max()
    elif len(maps.shape) == 4:  # [unit, yn, xn, 3]
        for unit_i in range(num_units):
            for color_i in range(3):
                if sigma == 0:
                    smoothed_maps[unit_i,:,:,color_i] = maps[unit_i,:,:,color_i]
                else:
                    smoothed_maps[unit_i,:,:,color_i] = gaussian_filter(maps[unit_i,:,:,color_i], sigma=sigma)
            if not math.isclose(smoothed_maps[unit_i].max(), 0, abs_tol=10 ** (-5)):
                smoothed_maps[unit_i] = smoothed_maps[unit_i]/smoothed_maps[unit_i].max()
    return smoothed_maps
    
def plot_r_val(r_val, font_size):
    plt.xticks([])
    plt.yticks([])
    plt.text(0.1, 0.7, f"r = {r_val:.4f}", fontsize=font_size, color='w')
    if math.isfinite(r_val):
        ax = plt.gca()
        ax.set_facecolor((r_val/2 + 0.5, 1 - abs(r_val), 1-r_val/2 - 0.5))

###############################################################################

# Set path to record r values in a txt file.
txt_path = os.path.join(result_dir, f"{max_or_min}_map_r_{sigma_rf_ratio:.4f}.txt")
if os.path.exists(txt_path):
    os.remove(txt_path)
# Give column names
with open(txt_path, 'a') as f:
    f.write(f"LAYER UNIT")
    for idx1, map_name1 in enumerate(all_map_names):
        for idx2, map_name2 in enumerate(all_map_names):
            if idx1 <= idx2:
                f.write(f" {map_name1}_vs_{map_name2}")
    f.write("\n")

########################### COMPUTE CORRELATIONS ##############################

for conv_i in range(num_layers):
    layer_name = f"conv{conv_i+1}"

    # Load the maps of this layer.
    # try:
    all_smoothed_maps = {}
    for map_name in all_map_names:
        this_map = load_maps(map_name, layer_name, max_or_min)
        sigma = this_map.shape[-2] * sigma_rf_ratio
        all_smoothed_maps[map_name] = smooth_and_normalize_maps(this_map, sigma)
    # except:
    #     break  # This layer was not mapped.
    
    num_units = all_smoothed_maps['gt'].shape[0]

    # Correlate the maps.
    for unit_i in tqdm(range(num_units)):
        # Display the r values and p values.
        r_vals = []
        for idx1, (name1, map1) in enumerate(all_smoothed_maps.items()):
            for idx2, (name2, map2) in enumerate(all_smoothed_maps.items()):
                if idx1 <= idx2:
                    unit_map1 = map1[unit_i]
                    unit_map2 = map2[unit_i]
                    # If the map has color, average the color channel.
                    if len(unit_map1.shape) == 3:
                        unit_map1 = np.mean(unit_map1, axis=2)
                    if len(unit_map2.shape) == 3:
                        unit_map2 = np.mean(unit_map2, axis=2)

                    # Compute correlations
                    r_val, p_val = pearsonr(unit_map1.flatten(), unit_map2.flatten())
                    r_vals.append(r_val)
        
        # Record correlations in text file
        with open(txt_path, 'a') as f:
            f.write(f"{layer_name} {unit_i}")
            for this_r_val in r_vals:
                f.write(f" {this_r_val:.4f}")
            f.write("\n")

################################ MAKE PDF #####################################

corr_df = pd.read_csv(txt_path, sep=' ', header=0)

if to_plot_pdf:
    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"
        
        # Initialize counter dictionary to record the number of 'good' map pairs.
        high_r_val_counts = {}
        for idx1, map_name1 in enumerate(all_map_names):
            for idx2, map_name2 in enumerate(all_map_names):
                if idx1 < idx2:
                    high_r_val_counts[f"{map_name1}_vs_{map_name2}"] = 0
        
        # Load the maps of this layer.
        try:
            all_smoothed_maps = {}
            for map_name in all_map_names:
                this_map = load_maps(map_name, layer_name, max_or_min)
                sigma = this_map.shape[-2] * sigma_rf_ratio
                all_smoothed_maps[map_name] = smooth_and_normalize_maps(this_map, sigma)
        except:
            break  # This layer was not mapped.

        pdf_path = os.path.join(result_dir, f"{layer_name}_{max_or_min}_map_r_{sigma_rf_ratio:.4f}.pdf")
        with PdfPages(pdf_path) as pdf:
            
            num_units = all_smoothed_maps['gt'].shape[0]

            # Correlate the maps.
            for unit_i in tqdm(range(num_units)):
                if this_is_a_test_run and unit_i > 10:
                    break

                plt.figure(figsize=(4*len(all_smoothed_maps), 4*len(all_smoothed_maps) - 2))
                plt.suptitle(f"Correlations of different maps (no.{unit_i}, {max_or_min})", fontsize=32)

                # Plot the maps at the margin.
                for idx, (map_name, map) in enumerate(all_smoothed_maps.items()):
                    plt.subplot(len(all_smoothed_maps)+1, len(all_smoothed_maps)+1, idx + 2)
                    plt.imshow(map[unit_i], cmap='gray')
                    plt.title(map_name, fontsize=font_size)

                    plt.subplot(len(all_smoothed_maps)+1, len(all_smoothed_maps)+1,
                                (len(all_smoothed_maps) + 1) * (idx + 1) + 1)
                    plt.imshow(map[unit_i], cmap='gray')
                    plt.title(map_name, fontsize=font_size)

                # Display the r values and p values.
                for idx1, name1 in enumerate(all_smoothed_maps.keys()):
                    for idx2, name2 in enumerate(all_smoothed_maps.keys()):
                        if idx1 <= idx2:
                            plt.subplot(len(all_smoothed_maps)+1,
                                        len(all_smoothed_maps)+1,
                                        (len(all_smoothed_maps) + 1) * (idx1 + 1) + 2 + idx2)
                            r_val = corr_df.loc[(corr_df.LAYER == layer_name) & (corr_df.UNIT == unit_i), f'{name1}_vs_{name2}'].item()
                            plot_r_val(r_val, font_size)
                            if idx1 != idx2 and r_val > r_val_threshold:
                                name1 = all_map_names[idx1]
                                name2 = all_map_names[idx2]
                                high_r_val_counts[f'{name1}_vs_{name2}'] += 1
                pdf.savefig()
                plt.close()

            plt.figure(figsize=(len(high_r_val_counts) * 3, 8))
            bars = plt.bar(high_r_val_counts.keys(), high_r_val_counts.values())
            plt.gca().bar_label(bars)   # Display the counts on top of the bars.
            plt.ylabel('counts', fontsize=font_size)
            plt.title(f"Distribution of r values higher than {r_val_threshold}", fontsize=font_size)
            pdf.savefig()
            plt.show()
            plt.close()
