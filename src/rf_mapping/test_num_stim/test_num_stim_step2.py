"""
Find the good number of bars used for our mapping methods.

Step 2. Gaussian fit.

Tony Fu, August 18th, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.gaussian_fit import (gaussian_fit,
                                        calc_f_explained_var,
                                        theta_to_ori)
from src.rf_mapping.gaussian_fit import GaussianFitParamFormat as ParamFormat
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
image_shape = (227, 227)
this_is_a_test_run = False
batch_size = 10
conv_i_to_run = 1  # conv_i = 1 means Conv2
rfmp_name = 'rfmp4c7o'
num_stim_list = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]

source_dir = os.path.join(c.REPO_DIR, 'results', 'test_num_stim')
result_dir = source_dir

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
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)

# Helper functions.
def write_txt(f, layer_name, unit_i, raw_params, fxvar, map_size, num_bars):
    # Unpack params
    amp = raw_params[ParamFormat.A_IDX]
    mu_x = raw_params[ParamFormat.MU_X_IDX]
    mu_y = raw_params[ParamFormat.MU_Y_IDX]
    sigma_1 = raw_params[ParamFormat.SIGMA_1_IDX]
    sigma_2 = raw_params[ParamFormat.SIGMA_2_IDX]
    theta = raw_params[ParamFormat.THETA_IDX]
    offset = raw_params[ParamFormat.OFFSET_IDX]
    
    # Some primitive processings:
    # (1) move original from top-left to map center.s
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
    f.write(f"{fxvar:.4f} ")  # the fraction of variance explained by params
    f.write(f"{num_bars}\n")


def load_maps(map_name, layer_name, num_stim, max_or_min):
    """Loads the maps of the layer."""
    
    if map_name == 'rfmp4a':
        mapping_path = os.path.join(source_dir,
                                    'rfmp4a',
                                    model_name,
                                    layer_name,
                                    str(num_stim),
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'rfmp4c7o':
        mapping_path = os.path.join(source_dir,
                                    'rfmp4c7o',
                                    model_name,
                                    layer_name,
                                    str(num_stim),
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.mean(maps, axis=1)
    elif map_name == 'rfmp_sin1':
        mapping_path = os.path.join(source_dir,
                                    'rfmp_sin1',
                                    model_name,
                                    layer_name,
                                    str(num_stim),
                                    f"{layer_name}_weighted_{max_or_min}_sinemaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'pasu':
        mapping_path = os.path.join(source_dir,
                                    'pasu',
                                    model_name,
                                    layer_name,
                                    str(num_stim),
                                    f"{layer_name}_weighted_{max_or_min}_shapemaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    else:
        raise KeyError(f"{map_name} does not exist.")


def geo_mean(sd1, sd2):
    return np.sqrt(np.power(sd1, 2) + np.power(sd2, 2))


layer_name = f"conv{conv_i_to_run + 1}"
rf_size = rf_sizes[conv_i_to_run][0]


for num_stim in num_stim_list:
    # Load bar counts:
    max_bar_counts = []
    min_bar_counts = []
    stim_counts_path = os.path.join(source_dir, rfmp_name, model_name, layer_name, str(num_stim), f"{model_name}_{rfmp_name}_weighted_counts.txt")
    with open(stim_counts_path) as count_f:
        count_lines = count_f.readlines()
        # Each line is made of: [layer_name unit num_max_bars num_min_bars]
        for line in count_lines:
            if line.split(' ')[0] == layer_name:
                max_bar_counts.append(int(line.split(' ')[2]))
                min_bar_counts.append(int(line.split(' ')[3]))

    # Load bar maps:
    max_maps = load_maps(rfmp_name, layer_name, num_stim, 'max')
    min_maps = load_maps(rfmp_name, layer_name, num_stim, 'min')
    
    top_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name, str(num_stim), f"gaussian_fit_weighted_top.txt")
    bot_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name, str(num_stim), f"gaussian_fit_weighted_bot.txt")
    
    # Delete previous files
    if os.path.exists(top_txt_path):
        os.remove(top_txt_path)
    if os.path.exists(bot_txt_path):
        os.remove(bot_txt_path)

    pdf_path = os.path.join(result_dir, rfmp_name, model_name, layer_name, str(num_stim),
                            f"{layer_name}_weighted_gaussian.pdf")
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
            with open(top_txt_path, 'a') as top_f:
                write_txt(top_f, layer_name, unit_i, params, fxvar, rf_size, max_bar_counts[unit_i])

            # plt.title(f"max (nbar = {max_bar_counts[unit_i]}, fxvar = {fxvar:.4f})\n"
            #         f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
            #         f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
            #         f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
            #         f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
            #         f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
            #         f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
            #         f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f})",
            #         fontsize=14)
            radius = geo_mean(params[ParamFormat.SIGMA_1_IDX], params[ParamFormat.SIGMA_2_IDX])
            plt.title(f"max {radius:.2f}", fontsize=18)

            plt.subplot(1, 2, 2)
            params, sems = gaussian_fit(min_map, plot=True, show=False)
            fxvar = calc_f_explained_var(min_map, params)
            with open(bot_txt_path, 'a') as bot_f:
                write_txt(bot_f, layer_name, unit_i, params, fxvar, rf_size, min_bar_counts[unit_i])
            # plt.title(f"min (nbar = {min_bar_counts[unit_i]}, fxvar = {fxvar:.4f})\n"
            #         f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
            #         f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
            #         f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
            #         f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
            #         f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
            #         f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
            #         f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f})",
            #         fontsize=14)
            radius = geo_mean(params[ParamFormat.SIGMA_1_IDX], params[ParamFormat.SIGMA_2_IDX])
            plt.title(f"min {radius:.2f}", fontsize=18)

            pdf.savefig()
            if this_is_a_test_run: plt.show()
            plt.close()
