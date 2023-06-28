"""
Compute the center of mass for all maps out there.

Tony Fu, Sep 19, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c
from src.rf_mapping.reproducibility import set_seeds


# Please specify some details here:
# set_seeds()
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"

this_is_a_test_run = False
is_random = False
map_name = 'gt'
sigma_rf_ratio = 1/30


###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n]") 
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)


#############################  HELPER FUNCTIONS  ##############################


def smooth_maps(maps, sigma):
    for i in range(maps.shape[0]):
            maps[i] = gaussian_filter(maps[i], sigma=sigma)
    return maps

def load_maps(map_name, layer_name, max_or_min, is_random, rf_size):
    """Loads the maps of the layer."""
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""
    sigma = sigma_rf_ratio * rf_size
    
    if map_name == 'gt':
        mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    f'backprop_sum{is_random_str}',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_{max_or_min}.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        return smooth_maps(maps, sigma)
    elif map_name == 'gt_composite':
        top_mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    f'backprop_sum{is_random_str}',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_max.npy")
        bot_mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    f'backprop_sum{is_random_str}',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_min.npy")
        top_maps = np.load(top_mapping_path)  # [unit, yn, xn]
        bot_maps = np.load(bot_mapping_path)  # [unit, yn, xn]
        return smooth_maps(top_maps + bot_maps, sigma)
    elif map_name == 'occlude':
        mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_{max_or_min}.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        return smooth_maps(maps, sigma)
    elif map_name == 'occlude_composite':
        top_mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_max.npy")
        bot_mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_min.npy")
        top_maps = np.load(top_mapping_path)  # [unit, yn, xn]
        bot_maps = np.load(bot_mapping_path)  # [unit, yn, xn]
        return smooth_maps(top_maps + bot_maps, sigma)
    elif map_name == 'rfmp4a':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4a',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        return smooth_maps(maps, sigma)
    elif map_name == 'rfmp4c7o':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4c7o',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        maps = np.mean(maps, axis=1)
        return smooth_maps(maps, sigma)
    elif map_name == 'rfmp_sin1':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp_sin1',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_sinemaps.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        return smooth_maps(maps, sigma)
    elif map_name == 'pasu':
        mapping_path = os.path.join(mapping_dir,
                                    'pasu',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_shapemaps.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        return smooth_maps(maps, sigma)
    elif map_name == 'block':
        mapping_path = os.path.join(mapping_dir,
                                    'block',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_blockmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        maps = np.mean(maps, axis=1)
        return smooth_maps(maps, sigma)
    else:
        raise KeyError(f"{map_name} does not exist.")


def get_result_dir(map_name, is_random, this_is_a_test_run):
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""

    if map_name in ('gt', 'gt_composite'):
        if this_is_a_test_run:
            return os.path.join(mapping_dir,
                                'ground_truth',
                                f'gaussian_fit{is_random_str}',
                                model_name,
                                'test')
        else:
            return os.path.join(mapping_dir,
                                'ground_truth',
                                f'gaussian_fit{is_random_str}',
                                model_name,
                                'abs')
    elif map_name in ('occlude', 'occlude_composite', 'rfmp4a', 'rfmp4c7o',
                      'rfmp_sin1', 'pasu', 'block'):
        if map_name == 'occlude_composite':
            map_name = 'occlude'
        
        if this_is_a_test_run:
            return os.path.join(mapping_dir,
                                map_name,
                                'gaussian_fit',
                                'test')
        else:
            return os.path.join(mapping_dir,
                                map_name,
                                'gaussian_fit',
                                model_name)
    else:
        raise KeyError(f"{map_name} does not exist.")


def write_txt(f, layer_name, unit_i, top_x, top_y, bot_x, bot_y, map_shape):
    # Center the origin
    top_y -= map_shape[0]/2
    top_x -= map_shape[1]/2
    bot_y -= map_shape[0]/2
    bot_x -= map_shape[1]/2
    f.write(f"{layer_name} {unit_i} {top_x} {top_y} {bot_x} {bot_y}\n")


def make_point(xc, yc, linewidth=1):
    circ = patches.Circle((xc, yc), 1, linewidth=linewidth,
                          edgecolor='r', facecolor='none')
    return circ


def write_pdf(pdf, layer_name, unit_i, top_map, bot_map,
              top_x, top_y, bot_x, bot_y):
    # Fit 2D Gaussian, and plot them.
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Center of mass ({layer_name} no.{unit_i})", fontsize=20)

    plt.subplot(1, 2, 1)
    plt.imshow(top_map, cmap='gray')
    plt.title(f"max\nmax_x = {top_x}, max_y = {top_y}", fontsize=14)
    ax = plt.gca()
    ax.add_patch(make_point(top_x, top_y))

    plt.subplot(1, 2, 2)
    plt.imshow(bot_map, cmap='gray')
    plt.title(f"min\nmax_x = {bot_x}, max_y = {bot_y}", fontsize=14)
    ax = plt.gca()
    ax.add_patch(make_point(bot_x, bot_y))

    pdf.savefig()
    if this_is_a_test_run: plt.show()
    plt.close()


def write_composite_pdf(pdf, layer_name, unit_i, composite_map, top_x, top_y):
    # Fit 2D Gaussian, and plot them.
    plt.figure(figsize=(8, 8))
    plt.title(f"Center of mass ({layer_name} no.{unit_i})\nx = {top_x}, y = {top_y}", fontsize=20)

    plt.imshow(composite_map, cmap='gray')
    ax = plt.gca()
    ax.add_patch(make_point(top_x, top_y))

    pdf.savefig()
    if this_is_a_test_run: plt.show()
    plt.close()


###############################################################################

result_dir = get_result_dir(map_name, is_random, this_is_a_test_run)
txt_file_path = os.path.join(result_dir, f"{model_name}_{map_name}_hot_spot.txt")

# Delete previous files.
if os.path.exists(txt_file_path):
    os.remove(txt_file_path)

# Find the center of mass coordinates and radius of RF.
for conv_i in range(len(layer_indices)):
    # if model_name == 'vgg16' and conv_i < 1:
    #     continue
    # Get layer-specific info
    layer_name = f"conv{conv_i + 1}"
    rf_size = rf_sizes[conv_i][0]

    # Load bar maps:
    if map_name not in ('gt_composite', 'occlude_composite'):
        max_maps = load_maps(map_name, layer_name, 'max', is_random, rf_size)
        min_maps = load_maps(map_name, layer_name, 'min', is_random, rf_size)

        pdf_path = os.path.join(result_dir, f"{layer_name}_hot_spot.pdf")
        with PdfPages(pdf_path) as pdf:
            for unit_i, (max_map, min_map) in enumerate(tqdm(zip(max_maps, min_maps))):
                # Do only the first 5 unit during testing phase
                if this_is_a_test_run and unit_i >= 5:
                    break
                
                top_y, top_x = np.unravel_index(np.argmax(max_map), max_map.shape)
                bot_y, bot_x = np.unravel_index(np.argmax(min_map), min_map.shape)

                with open(txt_file_path, 'a') as f:
                    write_txt(f, layer_name, unit_i, top_x, top_y, bot_x, bot_y, max_map.shape)

                # write_pdf(pdf, layer_name, unit_i, max_map, min_map,
                #         top_x, top_y, bot_x, bot_y)

    else:
        composite_maps = load_maps(map_name, layer_name, '', is_random, rf_size)

        pdf_path = os.path.join(result_dir, f"{layer_name}_composite_hot_spot.pdf")
        with PdfPages(pdf_path) as pdf:
            for unit_i, composite_map in enumerate(tqdm(composite_maps)):
                # Do only the first 5 unit during testing phase
                if this_is_a_test_run and unit_i >= 5:
                    break
                
                top_y, top_x = np.unravel_index(np.argmax(composite_map), composite_map.shape)

                with open(txt_file_path, 'a') as f:
                    # Fill the bot_x and bot_y columns with NaN
                    write_txt(f, layer_name, unit_i, top_x, top_y, np.NaN, np.NaN, composite_map.shape)

                # write_composite_pdf(pdf, layer_name, unit_i, composite_map, top_x, top_y)
