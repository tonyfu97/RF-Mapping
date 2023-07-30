"""
Script to summarize the Gaussian fit statistics.

Tony Fu, July 14, 2022
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
from gaussian_fit import GaussianFitParamFormat as ParamFormat
from hook import ConvUnitCounter
from spatial import get_rf_sizes
from mapping import RfMapper
import constants as c


# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model_name = 'alexnet'
cumulate_modes = ['or']
image_shape = (227, 227)
this_is_a_test_run = False

# Please double-check the directories:
fit_stat_dir = c.RESULTS_DIR + f'/rfmp4a/gaussian_fit/{model_name}'
pdf_dir = fit_stat_dir

###############################################################################

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

for cumulate_mode in cumulate_modes:
    for max_or_min in ['max', 'min', 'both']:
        fit_stat_dir_with_mode = os.path.join(fit_stat_dir, cumulate_mode)
        pdf_dir_with_mode = os.path.join(pdf_dir, cumulate_mode)
        pdf_path = os.path.join(pdf_dir_with_mode, f"fit_summary_{max_or_min}.pdf")
        all_params_sems = []

        with PdfPages(pdf_path) as pdf:
            # Load the fit parameters and SEMs of all layers.
            for conv_i, rf_size in enumerate(tqdm(rf_sizes)):     
                layer_name = f"conv{conv_i+1}"
                fit_stat_path = os.path.join(fit_stat_dir_with_mode, f"{layer_name}_{max_or_min}.npy")
                fit_params_sems = np.load(fit_stat_path)  # already cleaned
                all_params_sems.append(fit_params_sems)

            # Plot the parameters distributions:
            rf_sizes_1d = [rf_size[0] for rf_size in rf_sizes]
            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: elliptical axial length vs. maximum RF size ({max_or_min}, cumulate mode = {cumulate_mode})", fontsize=20)
            
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
            plt.suptitle(f"{model_name}: centers of elliptical Gaussians ({max_or_min}, cumulate mode = {cumulate_mode})", fontsize=20)
            for layer_i, layer_params_sems in enumerate(all_params_sems):
                bm = RfMapper(model, layer_i, image_shape)
                rf_size = rf_sizes_1d[layer_i]
                plt.subplot(1, num_layers, layer_i + 1)
                x_data = layer_params_sems[:, ParamFormat.MU_X_IDX, 0]
                y_data = layer_params_sems[:, ParamFormat.MU_Y_IDX, 0]
                plt.scatter(x_data, y_data, alpha=0.5)
                boundary = 5
                plt.xlim([bm.box[1] - boundary, bm.box[3] + boundary])
                plt.ylim([bm.box[0] - boundary, bm.box[2] + boundary])
                plt.xlabel("mu_x")
                plt.ylabel("mu_y")
                plt.title(f"conv{layer_i + 1} (n = {len(x_data)})")
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: orientations of receptive fields ({max_or_min}, cumulate mode = {cumulate_mode})", fontsize=20)
            for layer_i, layer_params_sems in enumerate(all_params_sems):
                plt.subplot(1, num_layers, layer_i + 1)
                orientations = layer_params_sems[:, ParamFormat.THETA_IDX, 0]
                plt.hist(orientations)
                plt.xlabel("orientation (degrees)")
                plt.ylabel("counts")
                plt.title(f"conv{layer_i + 1} (n = {len(orientations)})")
            pdf.savefig()
            plt.close()
