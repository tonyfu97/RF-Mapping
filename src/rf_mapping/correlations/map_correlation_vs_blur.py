"""
Script to correlate the maps and plot the distributions of r values.

Tony Fu, August 21st, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append('../../..')
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True)
# model_name = 'resnet18'

image_shape = (227, 227)
this_is_a_test_run = False
font_size = 20
sigma_rf_ratios = [0, 1/120, 1/60, 1/30, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2]
sigma_rf_ratios_str = ['0', '1/120', '1/60', '1/30', '1/20', '1/10', '1/5', '1/4', '1/3', '1/2']

# ADDING NEW MAP? MODIFY BELOW:
all_map_names = ['gt', 'gt_composite', 'occlude_composite',
                 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']

# File paths:
corr_path = os.path.join(c.RESULTS_DIR,
                         'compare',
                         'map_correlations',
                         model_name)
result_pdf_path = os.path.join(corr_path,
                               f"{model_name}_gaussian_blur_vs_correlations.pdf")

###############################################################################

# Get some layer info:
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

# Load correlations
all_max_layer_averages = []
all_min_layer_averages = []
for sigma_rf_ratio in sigma_rf_ratios:
    max_txt_path = os.path.join(corr_path, f"max_map_r_{sigma_rf_ratio:.4f}.txt")
    max_df = pd.read_csv(max_txt_path, sep=' ', header=0)
    all_max_layer_averages.append(max_df.groupby('LAYER', as_index=False).mean())
    
    min_txt_path = os.path.join(corr_path, f"min_map_r_{sigma_rf_ratio:.4f}.txt")
    min_df = pd.read_csv(min_txt_path, sep=' ', header=0)
    all_min_layer_averages.append(min_df.groupby('LAYER', as_index=False).mean())


# Make pdf:
with PdfPages(result_pdf_path) as pdf:
    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"

        plt.figure(figsize=(20, 20))
        plt.suptitle(f"{layer_name}", fontsize=24)
        for idx1, map_name1 in enumerate(all_map_names):
            for idx2, map_name2 in enumerate(all_map_names):
                if idx1 <= idx2:
                    corr_name = f"{map_name1}_vs_{map_name2}"
                    avg_corr = [df.loc[df.LAYER == layer_name, corr_name] for df in all_max_layer_averages]
                    
                    plt.subplot(len(all_map_names), len(all_map_names),
                                idx1 * len(all_map_names) + idx2 + 1)
                    plt.plot(sigma_rf_ratios, avg_corr, '.-')
                    # plt.xticks(sigma_rf_ratios, sigma_rf_ratios_str)
                    plt.ylim([0, 1.1])
                    
                    if idx1 == 0:
                        plt.title(map_name2, fontsize=18)
                    if idx1 == idx2:
                        plt.ylabel(map_name1, fontsize=18)
                    
        pdf.savefig()
        plt.show()
        plt.close()
