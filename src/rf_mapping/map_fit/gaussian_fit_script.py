"""
Gaussian fit script for all maps out there.

Tony Fu, Sep 19, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

sys.path.append('../../..')
from src.rf_mapping.gaussian_fit import (gaussian_fit,
                                        calc_f_explained_var,
                                        theta_to_ori)
from src.rf_mapping.gaussian_fit import GaussianFitParamFormat as ParamFormat
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.reproducibility import set_seeds
import src.rf_mapping.constants as c


# Please specify some details here:
set_seeds()
model = models.alexnet(pretrained=False).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"

this_is_a_test_run = True
is_random = False
map_name = 'rfmp_sin1'

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take time to run. Are you sure? [y/n] ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")


# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)

#############################  HELPER FUNCTIONS  ##############################

def write_txt(f, layer_name, unit_i, raw_params, explained_variance, map_size):
    # Unpack params
    amp = raw_params[ParamFormat.A_IDX]
    mu_x = raw_params[ParamFormat.MU_X_IDX]
    mu_y = raw_params[ParamFormat.MU_Y_IDX]
    sigma_1 = raw_params[ParamFormat.SIGMA_1_IDX]
    sigma_2 = raw_params[ParamFormat.SIGMA_2_IDX]
    theta = raw_params[ParamFormat.THETA_IDX]
    offset = raw_params[ParamFormat.OFFSET_IDX]
    
    # Some primitive processings:
    # (1) move original from top-left to map center.
    mu_x = mu_x - (map_size/2)
    mu_y = mu_y - (map_size/2)
    # (2) take the abs value of sigma values.
    sigma_1 = abs(sigma_1)
    sigma_2 = abs(sigma_2)
    # (3) convert theta to orientation.
    orientation = theta_to_ori(sigma_1, sigma_2, theta)

    f.write(f"{layer_name} {unit_i} ")
    f.write(f"{mu_x:.2f} {mu_y:.2f} ")
    f.write(f"{sigma_1:.2f} {sigma_2:.2f} ")
    f.write(f"{orientation:.2f} ")
    f.write(f"{amp:.3f} {offset:.3f} ")
    f.write(f"{explained_variance:.4f}\n")


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


###############################################################################

result_dir = get_result_dir(map_name, is_random, this_is_a_test_run)

top_file_path = os.path.join(result_dir, f"{model_name}_{map_name}_gaussian_top.txt")
bot_file_path = os.path.join(result_dir, f"{model_name}_{map_name}_gaussian_bot.txt")

# Delete previous files.
if os.path.exists(top_file_path):
    os.remove(top_file_path)
if os.path.exists(bot_file_path):
    os.remove(bot_file_path)

for conv_i in range(len(layer_indices)):
    layer_name = f"conv{conv_i + 1}"
    num_units = nums_units[conv_i]
    print(f"Fitting elliptical Gaussian for {layer_name}...")

    # For param_cleaner to check if Gaussian is inside in RF or not.
    rf_size = rf_sizes[conv_i][0]
    box = (0, 0, rf_size, rf_size)

    # Load backprop sums:
    max_maps = load_maps(map_name, layer_name, 'max', is_random, rf_size)
    min_maps = load_maps(map_name, layer_name, 'min', is_random, rf_size)

    pdf_path = os.path.join(result_dir, f"{layer_name}_gaussian_fit.pdf")
    with PdfPages(pdf_path) as pdf:
        
        for unit_i, (max_map, min_map) in enumerate(tqdm(zip(max_maps, min_maps))):
            # Do only the first 5 unit during testing phase
            if this_is_a_test_run and unit_i >= 5:
                break

            # Fit 2D Gaussian, and plot them.
            plt.figure(figsize=(20, 10))
            plt.suptitle(f"Elliptical Gaussian fit ({layer_name} no.{unit_i})", fontsize=20)

            plt.subplot(1, 2, 1)
            params, sems = gaussian_fit(max_map, plot=True, show=False)
            fxvar = calc_f_explained_var(max_map, params)
            with open(top_file_path, 'a') as top_f:
                write_txt(top_f, layer_name, unit_i, params, fxvar, rf_size)
            plt.title(f"max (fxvar = {fxvar:.4f})\n"
                        f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                        f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                        f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                        f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                        f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                        f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                        f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f})",
                        fontsize=14)

            plt.subplot(1, 2, 2)
            params, sems = gaussian_fit(min_map, plot=True, show=False)
            fxvar = calc_f_explained_var(min_map, params)
            with open(bot_file_path, 'a') as bot_f:
                write_txt(bot_f, layer_name, unit_i, params, fxvar, rf_size)
            plt.title(f"min (fxvar = {fxvar:.4f})\n"
                        f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                        f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                        f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                        f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                        f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                        f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                        f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f})",
                        fontsize=14)

            pdf.savefig()
            plt.close()
