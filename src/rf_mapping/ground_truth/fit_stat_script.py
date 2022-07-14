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

sys.path.append('..')
from hook import ConvUnitCounter
from spatial import get_rf_sizes
from gaussian_fit import ParamLoader
from mapping import RfMapper
import constants as c


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


for sum_mode in sum_modes:
    for max_or_min in ['max', 'min', 'both']:
        gaussian_fit_dir_with_mode = os.path.join(gaussian_fit_dir, sum_mode)
        pdf_path = os.path.join(gaussian_fit_dir_with_mode, f"summary_{max_or_min}.pdf")
        model_param_loaders = []
        num_layers = len(rf_sizes)
        
        with PdfPages(pdf_path) as pdf:
            for conv_i, rf_size in enumerate(tqdm(rf_sizes)):     
                layer_name = f"conv{conv_i + 1}"
                num_units = nums_units[conv_i]
                file_names = [f"{max_or_min}_{layer_name}.{i}.npy" for i in range(num_units)]
                mapper = RfMapper(model, conv_i, image_shape)
                param_loader = ParamLoader(gaussian_fit_dir_with_mode,
                                           file_names, mapper.box)
                model_param_loaders.append(param_loader)

            rf_sizes_1d = [rf_size[0] for rf_size in rf_sizes]
            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: elliptical axial length vs. maximum RF size ({max_or_min}, sum mode = {sum_mode})", fontsize=20)

            plt.subplot(1, 2, 1)
            sigma_1s = [param_loader.sigma_1s for param_loader in model_param_loaders]
            plt.boxplot(sigma_1s, positions=rf_sizes_1d, widths=np.full(len(rf_sizes),10))
            plt.xticks(rf_sizes_1d, [f"{rf_size}\n(conv{i+1})" for i, rf_size in enumerate(rf_sizes_1d)])
            plt.xlabel("RF size")
            plt.ylabel("sigma_1")

            plt.subplot(1, 2, 2)
            sigma_2s = [param_loader.sigma_2s for param_loader in model_param_loaders]
            plt.boxplot(sigma_2s, positions=rf_sizes_1d, widths=np.full(len(rf_sizes),10))
            plt.xticks(rf_sizes_1d, [f"{rf_size}\n(conv{i+1})" for i, rf_size in enumerate(rf_sizes_1d)])
            plt.xlabel("RF size")
            plt.ylabel("sigma_2")

            pdf.savefig()
            plt.close()


            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: centers of elliptical Gaussians ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            for layer_i, layer_param_loader in enumerate(model_param_loaders):
                rf_size = rf_sizes_1d[layer_i]
                plt.subplot(1, num_layers, layer_i + 1)
                plt.scatter(layer_param_loader.mu_xs, layer_param_loader.mu_ys, alpha=0.5)
                plt.xlim([0, rf_size])
                plt.ylim([0, rf_size])
                plt.xlabel("mu_x")
                plt.ylabel("mu_y")
                plt.title(f"conv{layer_i + 1} (n = {len(layer_param_loader.mu_xs)})")

            pdf.savefig()
            plt.close()

            plt.figure(figsize=(20,5))
            plt.suptitle(f"{model_name}: orientations of receptive fields ({max_or_min}, sum mode = {sum_mode})", fontsize=20)
            for layer_i, layer_param_loader in enumerate(model_param_loaders):
                plt.subplot(1, num_layers, layer_i + 1)
                plt.hist(layer_param_loader.orientations)
                plt.xlabel("orientation (degrees)")
                plt.ylabel("counts")
                plt.title(f"conv{layer_i + 1} (n = {len(layer_param_loader.orientations)})")

            pdf.savefig()
            plt.close()
