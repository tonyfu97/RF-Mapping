"""
Script to generate gaussian fit pdf and statistics for the visualization
results.

Tony Fu, June 29, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from gaussian_fit import gaussian_fit, ParamCleaner
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


for sum_mode in sum_modes:
    backprop_sum_dir_with_mode = os.path.join(backprop_sum_dir, sum_mode)
    result_dir_with_mode = os.path.join(result_dir, sum_mode)
    delete_all_npy_files(result_dir_with_mode)

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
                cleaned_params = param_cleaner.clean(params, sems, rf_sizes[conv_i])
                max_params_sems[unit_i, :, 0] = cleaned_params
                max_params_sems[unit_i, :, 1] = sems
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
                cleaned_params = param_cleaner.clean(params, sems, rf_sizes[conv_i])
                min_params_sems[unit_i, :, 0] = cleaned_params
                min_params_sems[unit_i, :, 1] = sems
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
                cleaned_params = param_cleaner.clean(params, sems, rf_sizes[conv_i])
                both_params_sems[unit_i, :, 0] = cleaned_params
                both_params_sems[unit_i, :, 1] = sems
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
