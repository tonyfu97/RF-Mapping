"""
Script to generate gaussian fit pdf and statistics for the visualization
results.

Tony Fu, June 29, 2022
"""
import os
import sys

import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from gaussian_fit import gaussian_fit
from gaussian_fit import GaussianFitParamFormat as ParamFormat
from hook import ConvUnitCounter
import constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True)
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

for sum_mode in sum_modes:
    backprop_sum_dir_with_mode = os.path.join(backprop_sum_dir, sum_mode)
    result_dir_with_mode = os.path.join(result_dir, sum_mode)

    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"

        # Load backprop sums:
        max_file_path = os.path.join(backprop_sum_dir_with_mode, f"max_{layer_name}.npy")
        min_file_path = os.path.join(backprop_sum_dir_with_mode, f"min_{layer_name}.npy")
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
                max_params_sems[unit_i, :, 0] = params
                max_params_sems[unit_i, :, 1] = sems
                plt.title(f"max\n"
                          f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                plt.subplot(1, 3, 2)
                params, sems = gaussian_fit(min_map, plot=True, show=False)
                min_params_sems[unit_i, :, 0] = params
                min_params_sems[unit_i, :, 1] = sems
                plt.title(f"min\n"
                          f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                plt.subplot(1, 3, 3)
                both_map = (max_map + min_map)/2
                params, sems = gaussian_fit(both_map, plot=True, show=False)
                both_params_sems[unit_i, :, 0] = params
                both_params_sems[unit_i, :, 1] = sems
                plt.title(f"max + min\n"
                          f"A={params[ParamFormat.A_IDX]:.2f}(err={sems[ParamFormat.A_IDX]:.2f}), "
                          f"mu_x={params[ParamFormat.MU_X_IDX]:.2f}(err={sems[ParamFormat.MU_X_IDX]:.2f}), "
                          f"mu_y={params[ParamFormat.MU_Y_IDX]:.2f}(err={sems[ParamFormat.MU_Y_IDX]:.2f}),\n"
                          f"sigma_1={params[ParamFormat.SIGMA_1_IDX]:.2f}(err={sems[ParamFormat.SIGMA_1_IDX]:.2f}), "
                          f"sigma_2={params[ParamFormat.SIGMA_2_IDX]:.2f}(err={sems[ParamFormat.SIGMA_2_IDX]:.2f}),\n"
                          f"theta={params[ParamFormat.THETA_IDX]:.2f}(err={sems[ParamFormat.THETA_IDX]:.2f}), "
                          f"offset={params[ParamFormat.OFFSET_IDX]:.2f}(err={sems[ParamFormat.OFFSET_IDX]:.2f}",
                          fontsize=14)

                pdf.savefig()
                plt.close()

        # Save fit parameters and SEMs:
        max_result_path = os.path.join(result_dir_with_mode, f"max_{layer_name}.npy")
        min_result_path = os.path.join(result_dir_with_mode, f"min_{layer_name}.npy")
        both_result_path = os.path.join(result_dir_with_mode, f"both_{layer_name}.npy")
        np.save(max_result_path, max_params_sems)
        np.save(min_result_path, min_params_sems)
        np.save(both_result_path, both_params_sems)
