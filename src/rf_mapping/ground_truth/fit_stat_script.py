"""
Code to summarize the statistics and visualize the trend of elliptical
Gaussian fit results.

Tony Fu, June 30, 2022
"""
import os
import sys

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.gaussian_fit import GaussianFitParamFormat as ParamFormat
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet().to(c.DEVICE)
model_name = "alexnet"
sum_modes = ['abs', 'sqr']
image_shape = (227, 227)

# Please double-check the directories:
gaussian_fit_dir = c.REPO_DIR + f'/results/ground_truth/gaussian_fit/{model_name}'
pdf_dir = gaussian_fit_dir

###############################################################################

# Get info of conv layers.
conv_counter = ConvUnitCounter(model)
_, nums_units = conv_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, nn.Conv2d)
num_layers = len(rf_sizes)

for sum_mode in sum_modes:
    for max_or_min in ['max', 'min', 'both']:
        gaussian_fit_dir_with_mode = os.path.join(gaussian_fit_dir, sum_mode)
        pdf_dir_with_mode = os.path.join(pdf_dir, sum_mode)
        pdf_path = os.path.join(pdf_dir_with_mode, f"fit_summary_{max_or_min}.pdf")
        all_params_sems = []

        with PdfPages(pdf_path) as pdf:
            # Load the fit parameters and SEMs of all layers.
            for conv_i, rf_size in enumerate(tqdm(rf_sizes)):     
                layer_name = f"conv{conv_i+1}"
                fit_stat_path = os.path.join(gaussian_fit_dir_with_mode, f"{layer_name}_{max_or_min}.npy")
                fit_params_sems = np.load(fit_stat_path)  # already cleaned
                all_params_sems.append(fit_params_sems)

            # Plot the parameters distributions:
            rf_sizes_1d = [rf_size[0] for rf_size in rf_sizes]
            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: elliptical axial length vs. maximum RF size ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            
            sigma_1s = []
            sigma_2s = []
            for fit_params_sems in all_params_sems:
                data_1 = fit_params_sems[:, ParamFormat.SIGMA_1_IDX, 0]
                data_2 = fit_params_sems[:, ParamFormat.SIGMA_2_IDX, 0]
                sigma_1s.append(data_1[np.isfinite(data_1)])
                sigma_2s.append(data_2[np.isfinite(data_2)])

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


            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: centers of elliptical Gaussians ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            for layer_i, layer_params_sems in enumerate(all_params_sems):
                rf_size = rf_sizes_1d[layer_i]
                plt.subplot(1, num_layers, layer_i + 1)
                x_data = layer_params_sems[:, ParamFormat.MU_X_IDX, 0]
                x_data = x_data[np.isfinite(x_data)]
                y_data = layer_params_sems[:, ParamFormat.MU_Y_IDX, 0]
                y_data = y_data[np.isfinite(y_data)]
                plt.scatter(x_data, y_data, alpha=0.5)
                plt.xlim([0, rf_size])
                plt.ylim([0, rf_size])
                plt.xlabel("mu_x")
                plt.ylabel("mu_y")
                plt.title(f"conv{layer_i + 1} (n = {len(x_data)})")
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: orientations of receptive fields ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            for layer_i, layer_params_sems in enumerate(all_params_sems):
                plt.subplot(1, num_layers, layer_i + 1)
                orientations = layer_params_sems[:, ParamFormat.THETA_IDX, 0]
                orientations = orientations[np.isfinite(orientations)]
                plt.hist(orientations)
                plt.xlabel("orientation (degrees)")
                plt.ylabel("counts")
                plt.title(f"conv{layer_i + 1} (n = {len(orientations)})")

            pdf.savefig()
            plt.close()
