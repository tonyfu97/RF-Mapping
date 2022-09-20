"""
Compute the center of mass for all maps out there.

Tony Fu, Sep 19, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.bar import mapstat_comr_1
import src.rf_mapping.constants as c
from src.rf_mapping.reproducibility import set_seeds


# Please specify some details here:
set_seeds()
model = models.alexnet(pretrained=False).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"

this_is_a_test_run = False
is_random = False
map_name = 'pasu'


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


def load_maps(map_name, layer_name, max_or_min, is_random, rf_size):
    """Loads the maps of the layer."""
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""
    
    if map_name == 'gt':
        mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    f'backprop_sum{is_random_str}',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_{max_or_min}.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'occlude':
        mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_{max_or_min}.npy")
        maps = np.load(mapping_path)  # [unit, yn, xn]
        # blur the maps
        sigma = rf_size / 60    # Reason behind the choice of sigma:
                                # The occlude stride is rf_size // 10 // 2
                                # so about rf_size / 20. A Gaussian is about
                                # 3 sigmas to each side => rf_size / 20 / 3
                                # = rf_size / 60.
        for i in range(maps.shape[0]):
            maps[i] = gaussian_filter(maps[i], sigma=sigma)
        return maps
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
        return np.mean(maps, axis=1)  # Need the color channel for plots.
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


def get_result_dir(map_name, is_random, this_is_a_test_run):
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""

    if map_name == 'gt':
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
    elif map_name in ('occlude', 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu'):
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


def write_txt(f, layer_name, unit_i,
              top_x, top_y, top_rad_10, top_rad_50, top_rad_90,
              bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90):
    # Center the origin
    top_y -= max_map.shape[0]/2
    top_x -= max_map.shape[1]/2
    bot_y -= min_map.shape[0]/2
    bot_x -= min_map.shape[1]/2
    
    f.write(f"{layer_name} {unit_i} ")
    f.write(f"{top_x:.2f} {top_y:.2f} {top_rad_10:.2f} {top_rad_50:.2f} {top_rad_90:.2f} ")
    f.write(f"{bot_x:.2f} {bot_y:.2f} {bot_rad_10:.2f} {bot_rad_50:.2f} {bot_rad_90:.2f}\n")


def make_circle(xc, yc, radius, linewidth=1):
    circ = patches.Circle((xc, yc), radius, linewidth=linewidth,
                          edgecolor='r', facecolor='none')
    return circ


def write_pdf(pdf, layer_name, unit_i, top_map, bot_map,
              top_x, top_y, top_rad_10, top_rad_50, top_rad_90,
              bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90):
    # Fit 2D Gaussian, and plot them.
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Center of mass ({layer_name} no.{unit_i})", fontsize=20)

    plt.subplot(1, 2, 1)
    plt.imshow(top_map, cmap='gray')
    plt.title(f"max\n"
              f"com_x = {top_x:.2f}, com_y = {top_y:.2f}\n"
              f"radius: {top_rad_10:.2f} (10%), {top_rad_50:.2f} (50%), {top_rad_90:.2f} (90%)",
              fontsize=14)
    ax = plt.gca()
    ax.add_patch(make_circle(top_x, top_y, top_rad_10))
    ax.add_patch(make_circle(top_x, top_y, top_rad_50))
    ax.add_patch(make_circle(top_x, top_y, top_rad_90))

    plt.subplot(1, 2, 2)
    plt.imshow(bot_map, cmap='gray')
    plt.title(f"min\n"
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


###############################################################################

result_dir = get_result_dir(map_name, is_random, this_is_a_test_run)
txt_file_path = os.path.join(result_dir, f"{model_name}_{map_name}_com.txt")

# Delete previous files.
if os.path.exists(txt_file_path):
    os.remove(txt_file_path)

# Find the center of mass coordinates and radius of RF.
for conv_i in range(len(layer_indices)):
    if model_name == 'vgg16' and conv_i < 1:
        continue
    # Get layer-specific info
    layer_name = f"conv{conv_i + 1}"
    rf_size = rf_sizes[conv_i][0]

    # Load bar maps:
    max_maps = load_maps(map_name, layer_name, 'max', is_random, rf_size)
    min_maps = load_maps(map_name, layer_name, 'min', is_random, rf_size)

    pdf_path = os.path.join(result_dir, f"{layer_name}_com.pdf")
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

            with open(txt_file_path, 'a') as f:
                write_txt(f, layer_name, unit_i,
                        top_x, top_y, top_rad_10, top_rad_50, top_rad_90,
                        bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90)

            write_pdf(pdf, layer_name, unit_i, max_map, min_map,
                top_x, top_y, top_rad_10, top_rad_50, top_rad_90,
                bot_x, bot_y, bot_rad_10, bot_rad_50, bot_rad_90)
