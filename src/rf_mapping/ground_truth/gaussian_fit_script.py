"""
Script to generate gaussian fit pdf and statistics for the visualization
results.

Tony Fu, June 29, 2022
"""
from cmath import exp
import os
import sys

from statistics import variance
import this
import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from gaussian_fit import gaussian_fit, ParamCleaner, twoD_Gaussian
from gaussian_fit import GaussianFitParamFormat as ParamFormat
from hook import ConvUnitCounter
from spatial import get_rf_sizes
from files import delete_all_npy_files
import constants as c


# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model_name = 'alexnet'
sum_modes = ['abs', 'sqr']
this_is_a_test_run = False

# Please double-check the directories:
backprop_sum_dir = c.REPO_DIR + f'/results/ground_truth/backprop_sum/{model_name}'
result_dir = c.REPO_DIR + f'/results/ground_truth/gaussian_fit/{model_name}'

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input(
        "This code may take time to run. Are you sure? "\
        f"All .npy files in {result_dir} will be deleted. (y/n): ") 
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")


# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)

# Helper objects:
param_cleaner = ParamCleaner()

# Helper functions for txt files:
def calc_explained_variance(gt_map, params):
    # Reconstruct map with fit parameters.
    x_size = gt_map.shape[1]
    y_size = gt_map.shape[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)
    fit_map = twoD_Gaussian((x, y),
                            params[ParamFormat.A_IDX],
                            params[ParamFormat.MU_X_IDX],
                            params[ParamFormat.MU_Y_IDX],
                            params[ParamFormat.SIGMA_1_IDX],
                            params[ParamFormat.SIGMA_2_IDX],
                            params[ParamFormat.THETA_IDX],
                            params[ParamFormat.OFFSET_IDX])
    # Calcualte variances
    residual_var = variance(fit_map - gt_map.flatten())
    gt_var = variance(gt_map.flatten())
    return 1 - (residual_var/gt_var)

def wrap_angle_180(angle):
    while angle >= 180:
        angle -= 180
    while angle < 0:
        angle += 180
    return angle

def theta_to_ori(sigma_1, sigma_2, theta):
    """
    Translates theta into orientation. Needs this function because theta
    tells us the orientation of sigma_1, which may or may not be the semi-
    major axis, whereas orientation is always about the semi-major axis.
    Therefore, when sigma_2 > sigma_1, our theta is off by 90 degrees from
    the actual orientation.

    Parameters
    ----------
    sigma_1 & sigma_2 : float
        The std. dev.'s of the semi-major and -minor axes. The larger of
        of the two is the semi-major.
    theta : float
        The orientation of sigma_1 in degrees.

    Returns
    -------
    orientation: float
        The orientation of the unit's receptive field in degrees.
    """
    if sigma_1 > sigma_2:
        return wrap_angle_180(theta)
    return wrap_angle_180(theta - 90)

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

    f.write(f"{layer_name:6} {unit_i:3} ")
    f.write(f"{mu_x:7.2f} {mu_y:7.2f} ")
    f.write(f"{sigma_1:7.2f} {sigma_2:7.2f} ")
    f.write(f"{orientation:6.2f} ")
    f.write(f"{amp:8.3f} {offset:8.3f} ")
    f.write(f"{explained_variance:7.4f}\n")


for sum_mode in sum_modes:
    backprop_sum_dir_with_mode = os.path.join(backprop_sum_dir, sum_mode)

    if this_is_a_test_run:
        result_dir_with_mode = os.path.join(result_dir, 'test')
    else:
        result_dir_with_mode = os.path.join(result_dir, sum_mode)
    
    # Delete previous files.
    delete_all_npy_files(result_dir_with_mode)
    top_file_path = os.path.join(result_dir_with_mode, f"{model_name}_gt_gaussian_top.txt")
    bot_file_path = os.path.join(result_dir_with_mode, f"{model_name}_gt_gaussian_bot.txt")
    if os.path.exists(top_file_path):
        os.remove(top_file_path)
    if os.path.exists(bot_file_path):
        os.remove(bot_file_path)

    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"

        # Load backprop sums:
        max_file_path = os.path.join(backprop_sum_dir_with_mode, f"{layer_name}_max.npy")
        min_file_path = os.path.join(backprop_sum_dir_with_mode, f"{layer_name}_min.npy")
        max_maps = np.load(max_file_path)  # [unit, y, x]
        min_maps = np.load(min_file_path)  # [unit, y, x]

        # Initialize arrays for parameters and standard error (SEM) values:
        num_units = nums_units[conv_i]
        num_params = ParamFormat.NUM_PARAMS
        max_params_sems = np.zeros((num_units, num_params, 2))
        min_params_sems = np.zeros((num_units, num_params, 2))
        both_params_sems = np.zeros((num_units, num_params, 2))
        
        # For param_cleaner to check if Gaussian is inside in RF or not.
        rf_size = rf_sizes[conv_i][0]
        box = (0, 0, rf_size, rf_size)

        pdf_path = os.path.join(result_dir_with_mode, f"{layer_name}.pdf")
        with PdfPages(pdf_path) as pdf:
            for unit_i, (max_map, min_map) in enumerate(tqdm(zip(max_maps, min_maps))):
                # Do only the first 5 unit during testing phase
                if this_is_a_test_run and unit_i >= 5:
                    break

                # Fit 2D Gaussian, and plot them.
                plt.figure(figsize=(30, 10))
                plt.suptitle(f"Elliptical Gaussian fit ({layer_name} no.{unit_i}, "
                             f"sum mode: {sum_mode})", fontsize=20)

                plt.subplot(1, 3, 1)
                params, sems = gaussian_fit(max_map, plot=True, show=False)
                exp_var = calc_explained_variance(max_map, params)
                with open(top_file_path, 'a') as f:
                    write_txt(f, layer_name, unit_i, params, exp_var, rf_size)
                cleaned_params = param_cleaner.clean(params, sems, box)
                max_params_sems[unit_i, :, 0] = cleaned_params
                max_params_sems[unit_i, :, 1] = sems
                if cleaned_params is None:  
                    cleaned_params = params
                plt.title(f"max\n"
                          f"A={cleaned_params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={cleaned_params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={cleaned_params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={cleaned_params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={cleaned_params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={cleaned_params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={cleaned_params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                plt.subplot(1, 3, 2)
                params, sems = gaussian_fit(min_map, plot=True, show=False)
                cleaned_params = param_cleaner.clean(params, sems, box)
                min_params_sems[unit_i, :, 0] = cleaned_params
                min_params_sems[unit_i, :, 1] = sems
                if cleaned_params is None:  
                    cleaned_params = params
                plt.title(f"min\n"
                          f"A={cleaned_params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={cleaned_params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={cleaned_params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={cleaned_params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={cleaned_params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={cleaned_params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={cleaned_params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                plt.subplot(1, 3, 3)
                both_map = (max_map + min_map)/2
                params, sems = gaussian_fit(both_map, plot=True, show=False)
                cleaned_params = param_cleaner.clean(params, sems, box)
                both_params_sems[unit_i, :, 0] = cleaned_params
                both_params_sems[unit_i, :, 1] = sems
                if cleaned_params is None:  
                    cleaned_params = params
                plt.title(f"max + min\n"
                          f"A={cleaned_params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={cleaned_params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={cleaned_params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={cleaned_params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={cleaned_params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={cleaned_params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={cleaned_params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                pdf.savefig()
                plt.close()

        # Save fit parameters and SEMs:
        max_result_path = os.path.join(result_dir_with_mode, f"{layer_name}_max.npy")
        min_result_path = os.path.join(result_dir_with_mode, f"{layer_name}_min.npy")
        both_result_path = os.path.join(result_dir_with_mode, f"{layer_name}_both.npy")
        np.save(max_result_path, max_params_sems)
        np.save(min_result_path, min_params_sems)
        np.save(both_result_path, both_params_sems)
