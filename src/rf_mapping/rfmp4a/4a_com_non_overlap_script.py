"""
Script to generate center of mass stats for the non-overlapping sums of the top
and bottom bars.

Tony Fu, July 21, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches


sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.bar import mapstat_comr_1
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)  # repeat for correct bar_count
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = 'vgg16'
image_shape = (227, 227)
this_is_a_test_run = False

# Source paths:
mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', model_name)
bar_counts_path = os.path.join(mapping_dir, f"{model_name}_rfmp4a_non_overlap_counts.txt")

# Result paths:
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit', model_name)
txt_path = os.path.join(result_dir, f"non_overlap.txt")

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take time to run. Are you sure? [y/n]") 
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Delete previous files
if os.path.exists(txt_path):
    os.remove(txt_path)

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)

# Helper functions.
def write_txt(f, layer_name, unit_i,
              top_x, top_y, top_rad_10, top_rad_50, top_rad_90, top_num_bars,
              bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90, bot_num_bars):
    # Center the origin
    top_y -= max_map.shape[0]/2
    top_x -= max_map.shape[1]/2
    bot_y -= min_map.shape[0]/2
    bot_x -= min_map.shape[1]/2
    
    f.write(f"{layer_name} {unit_i} ")
    f.write(f"{top_x:.2f} {top_y:.2f} {top_rad_10:.2f} {top_rad_50:.2f} {top_rad_90:.2f} {top_num_bars} ")
    f.write(f"{bot_x:.2f} {bot_y:.2f} {bot_rad_10:.2f} {bot_rad_50:.2f} {bot_rad_90:.2f} {bot_num_bars}\n")
    
def make_circle(xc, yc, radius, linewidth=1):
    circ = patches.Circle((xc, yc), radius, linewidth=linewidth,
                          edgecolor='r', facecolor='none')
    return circ

def write_pdf(pdf, layer_name, unit_i, top_map, bot_map,
              top_x, top_y, top_rad_10, top_rad_50, top_rad_90, top_num_bars,
              bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90, bot_num_bars):
    # Fit 2D Gaussian, and plot them.
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Center of mass ({layer_name} no.{unit_i})", fontsize=20)

    plt.subplot(1, 2, 1)
    plt.imshow(top_map, cmap='gray')
    plt.title(f"max (nbar = {top_num_bars})\n"
              f"com_x = {top_x:.2f}, com_y = {top_y:.2f}\n"
              f"radius: {top_rad_10:.2f} (10%), {top_rad_50:.2f} (50%), {top_rad_90:.2f} (90%)",
              fontsize=14)
    ax = plt.gca()
    ax.add_patch(make_circle(top_x, top_y, top_rad_10))
    ax.add_patch(make_circle(top_x, top_y, top_rad_50))
    ax.add_patch(make_circle(top_x, top_y, top_rad_90))

    plt.subplot(1, 2, 2)
    plt.imshow(bot_map, cmap='gray')
    plt.title(f"min (nbar = {bot_num_bars})\n"
              f"com_x = {bot_x:.2f}, com_y = {bot_y:.2f}\n"
              f"radius: {bot_rad_10:.2f} (10%), {bot_rad_50:.2f} (50%), {bot_rad_90:.2f} (90%)",
              fontsize=14)
    ax = plt.gca()
    ax.add_patch(make_circle(bot_x, bot_y, bot_rad_10))
    ax.add_patch(make_circle(bot_x, bot_y, bot_rad_50))
    ax.add_patch(make_circle(bot_x, bot_y, bot_rad_90))

    pdf.savefig()
    if this_is_a_test_run: plt.show()
    plt.close()


# Find the center of mass coordinates and radius of RF.
for conv_i in range(len(layer_indices)):
    if model_name == 'vgg16' and conv_i < 4:
        continue
    # Get layer-specific info
    layer_name = f"conv{conv_i + 1}"
    rf_size = rf_sizes[conv_i][0]
    
    # Load bar counts:
    max_bar_counts = []
    min_bar_counts = []
    with open(bar_counts_path) as count_f:
        count_lines = count_f.readlines()
        # Each line is made of: [layer_name unit num_max_bars num_min_bars]
        for line in count_lines:
            if line.split(' ')[0] == layer_name:
                max_bar_counts.append(int(line.split(' ')[2]))
                min_bar_counts.append(int(line.split(' ')[3]))

    # Load bar maps:
    max_maps_path = os.path.join(mapping_dir, f"{layer_name}_non_overlap_max_barmaps.npy")
    min_maps_path = os.path.join(mapping_dir, f"{layer_name}_non_overlap_min_barmaps.npy")
    max_maps = np.load(max_maps_path)
    min_maps = np.load(min_maps_path)


    pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_com.pdf")
    with PdfPages(pdf_path) as pdf:
        for unit_i, (max_map, min_map) in enumerate(tqdm(zip(max_maps, min_maps))):
            # Do only the first 5 unit during testing phase
            if this_is_a_test_run and unit_i >= 5:
                break
            
            top_y, top_x, top_rad_10 = mapstat_comr_1(max_map, 0.1)
            _, _, top_rad_50 = mapstat_comr_1(max_map, 0.5)
            _, _, top_rad_90 = mapstat_comr_1(max_map, 0.9)

            bot_y, bot_x, bot_rad_10 = mapstat_comr_1(min_map, 0.1)
            _, _, bot_rad_50 = mapstat_comr_1(min_map, 0.5)
            _, _, bot_rad_90 = mapstat_comr_1(min_map, 0.9)

            with open(txt_path, 'a') as f:
                write_txt(f, layer_name, unit_i,
                        top_x, top_y, top_rad_10, top_rad_50, top_rad_90, max_bar_counts[unit_i],
                        bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90, min_bar_counts[unit_i])

            write_pdf(pdf, layer_name, unit_i, max_map, min_map,
                top_x, top_y, top_rad_10, top_rad_50, top_rad_90, max_bar_counts[unit_i],
                bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90, min_bar_counts[unit_i])