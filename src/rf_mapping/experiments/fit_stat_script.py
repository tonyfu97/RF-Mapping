"""


Tony Fu, June 30, 2022
"""
import os
import sys
import warnings

import numpy as np
from pathlib import Path
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from hook import ConvUnitCounter, get_rf_sizes
from image import preprocess_img_for_plot


# Please specify some details here:
model = models.alexnet()
model_name = "alexnet"
sum_modes = ['abs', 'sqr']
this_is_a_test_run = False

# Please double-check the directories:
gaussian_fit_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/ground_truth/gaussian_fit/{model_name}')
pdf_dir = gaussian_fit_dir

###############################################################################

# Get info of conv layers.
conv_counter = ConvUnitCounter(model)
_, nums_units = conv_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), nn.Conv2d)


def wrap_angles_180(angles):
    """Makes sure all angles in an array is 0 <= angle < 180 degrees."""
    wrapped_angles = np.zeros(len(angles))

    for i, angle in enumerate(angles):
        while angle >= 180:
            angle -= 180
        while angle < 0:
            angle += 180
        wrapped_angles[i] = angle

    return wrapped_angles


def filter_params(params, sems, rf_size, thres=1):
    """
    Removes the units with poor Gaussian fits, i.e., those with at least one
    following characteristics:
        (1) Have at least one parameter with a SEM greater than the threshold.
        (2) Have at least one parameter that is equal to -1 (indicates the fit
            did not converge).
        (3) Have a center (mu_y, mu_x) that is outside of receptive field.
    
    Paramters
    ---------
    params : np.array
        [num_units, 7]. Each row corresponds to a unit.
    sem : np.array
        Same dimension as params. The standard errors of the means (SEMs) of
        the parameters.

    Returns
    -------
    filtered_params : np.array
        Just like params, but without the poorly fitted units.
    """
    num_units = params.shape[0]
    filtered_params = []

    for unit_i in range(num_units):
        mu_x = params[unit_i, 1]
        mu_y = params[unit_i, 2]
        unit_params = params[unit_i, :]
        unit_sems = sems[unit_i, :-2] # Don't care about angle and offset

        if np.all(unit_sems < thres) and\
           np.all(unit_params != -1) and\
           0 < mu_y < rf_size[0] and 0 < mu_x < rf_size[1]:
            filtered_params.append(np.absolute(unit_params))

    return np.array(filtered_params)


for sum_mode in sum_modes:
    for max_or_min in ['max', 'min', 'both']:
        gaussian_fit_dir_with_mode = os.path.join(gaussian_fit_dir, sum_mode)
        pdf_path = os.path.join(gaussian_fit_dir_with_mode, f"{max_or_min}.pdf")
        model_params = []
        
        with PdfPages(pdf_path) as pdf:
            for conv_i, rf_size in enumerate(tqdm(rf_sizes)):
                layer_name = f"conv{conv_i + 1}"
                num_units = nums_units[conv_i]

                layer_params = np.zeros((num_units, 7))
                layer_sems = np.zeros((num_units, 7))

                for unit_i in range(num_units):
                    # Do only the first 5 unit during testing phase
                    if this_is_a_test_run and unit_i >= 5:
                        break

                    param_sem_path = os.path.join(gaussian_fit_dir_with_mode,
                                                    f"{max_or_min}_{layer_name}.{unit_i}.npy")
                    params_sems = np.load(param_sem_path)
                    layer_params[unit_i, :] = params_sems[0, :]
                    layer_sems[unit_i, :] = params_sems[1, :]

                layer_params = filter_params(layer_params, layer_sems, rf_size)
                model_params.append(layer_params)

                plt.figure(figsize=(15,5))
                plt.suptitle(f"{model_name} elliptical fit summary for {layer_name} ({max_or_min}, n = {num_units}, sum mode: {sum_mode})", fontsize=20)

                plt.subplot(1, 3, 1)
                plt.scatter(layer_params[:, 2], layer_params[:, 1])
                plt.xlim([0, rf_size[1]])
                plt.ylim([0, rf_size[0]])
                plt.xlabel("mu_x")
                plt.ylabel("mu_y")

                plt.subplot(1, 3, 2)
                plt.hist(layer_params[:, 0])
                plt.xlabel("A")

                plt.subplot(1, 3, 3)
                plt.scatter(layer_params[:, 4], layer_params[:, 3])
                plt.xlim([0, rf_size[1]//2])
                plt.ylim([0, rf_size[0]//2])
                plt.xlabel("sigma_2")
                plt.ylabel("sigma_1")

                # plt.subplot(1, 5, 4)
                # plt.hist(wrap_angles_180(layer_params[:, 5]))
                # plt.xlabel("theta")

                # plt.subplot(1, 5, 5)
                # plt.hist(layer_params[:, 6])
                # plt.xlabel("offset")

                pdf.savefig()
                plt.close()


            sigma_1s = [layer_params[:, 3] for layer_params in model_params]
            sigma_2s = [layer_params[:, 4] for layer_params in model_params]
            rf_sizes_1d = [rf_size[0] for rf_size in rf_sizes]
            
            # TODO: ask Dr. Bair about 2 things:
            # (1) How to combine sigma 1 and 2?
            # (2) How to translate sigma values into RF field size?
            
            plt.figure(figsize=(15,5))
            plt.suptitle(f"{model_name} elliptical axial length vs. maximum RF size ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            
            plt.subplot(1, 2, 1)
            plt.boxplot(sigma_1s, positions=rf_sizes_1d, widths=np.full(len(rf_sizes),10))
            plt.xticks(rf_sizes_1d, [f"{rf_size}\n(conv{i+1})" for i, rf_size in enumerate(rf_sizes_1d)])
            plt.xlabel("RF size")
            plt.ylabel("sigma_1")
            
            plt.subplot(1, 2, 2)
            plt.boxplot(sigma_2s, positions=rf_sizes_1d, widths=np.full(len(rf_sizes),10))
            plt.xticks(rf_sizes_1d, [f"{rf_size}\n(conv{i+1})" for i, rf_size in enumerate(rf_sizes_1d)])
            plt.xlabel("RF size")
            plt.ylabel("sigma_2")
            
            pdf.savefig()
            plt.close()
