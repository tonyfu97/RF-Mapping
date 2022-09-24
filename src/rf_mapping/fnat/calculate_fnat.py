"""
Code to compute the Color Rotation Indices (CRI) of the units.

Tony Fu, August 19, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import pearsonr
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.result_txt_format import (CenterReponses as CR)
from src.rf_mapping.hook import ConvUnitCounter

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"

top_n_r = 1
this_is_a_test_run = True
rfmp_name = 'rfmp4a'

# Please double-check the directories:
gt_response_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', model_name)
rfmp_response_dir = os.path.join(c.REPO_DIR, 'results', rfmp_name, 'mapping', model_name)
result_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'fnat', rfmp_name, model_name)

###############################################################################

# Script guard.
if __name__ == "__main__":
    user_input = input("This code takes time to run. Are you sure? "\
                       "Enter 'y' to proceed. Type any other key to stop: ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")


# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)
num_layers = len(rf_sizes)


def config_plot(limits):
    line = np.linspace(min(limits), max(limits), 100)
    plt.plot(line, line, 'k', alpha=0.4)
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')


pdf_path = os.path.join(result_dir, f"{top_n_r}_avg_fnat.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(num_layers*5, 10))
    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"

        # Load Rfmp4 center responses
        top_rfmp_path = os.path.join(rfmp_response_dir, f'{layer_name}_top5000_responses.txt')
        top_rfmp_df = pd.read_csv(top_rfmp_path, sep=" ", header=None)
        top_rfmp_df.columns = [e.name for e in CR]

        bot_rfmp_path = os.path.join(rfmp_response_dir, f'{layer_name}_bot5000_responses.txt')
        bot_rfmp_df = pd.read_csv(bot_rfmp_path, sep=" ", header=None)
        bot_rfmp_df.columns = [e.name for e in CR]
        
        # Average the top- and bottom-N resposnes for each unit
        top_rfmp_responses = top_rfmp_df.loc[(top_rfmp_df.RANK < top_n_r), ['UNIT', 'R']]
        avg_top_rfmp_responses = top_rfmp_responses.groupby('UNIT').mean()
        
        bot_rfmp_responses = bot_rfmp_df.loc[(bot_rfmp_df.RANK < top_n_r), ['UNIT', 'R']]
        avg_bot_rfmp_responses = bot_rfmp_responses.groupby('UNIT').mean()
        
        # Load the GT responses.
        gt_response_path = os.path.join(gt_response_dir, f"{layer_name}_responses.npy")
        gt_responses = np.load(gt_response_path)
        # Shape = [num_images, num_units, 2]. There are 2 columns:
        # 0. Max responses of the given image and unit
        # 1. Min responses of the given image and unit
        
        # Average the top- and bottom-N responses for each unit
        avg_top_gt_responses = np.mean(gt_responses[:top_n_r, :, 0], axis=0)
        avg_bot_gt_responses = np.mean(gt_responses[:top_n_r, :, 1], axis=0)
        
        plt.subplot(2, num_layers, conv_i+1)
        plt.scatter(avg_top_gt_responses, avg_top_rfmp_responses)
        config_plot((-100,100))
        plt.xlabel(f"GT")
        plt.ylabel(f"{rfmp_name}")
        r_val, _ = pearsonr(avg_top_gt_responses, avg_top_rfmp_responses)
        plt.title(f"{layer_name}, r = {r_val:.4f}")

        plt.subplot(2, num_layers, conv_i+1+num_layers)
        plt.scatter(avg_bot_gt_responses, avg_bot_rfmp_responses)
        config_plot((-100,100))
        plt.xlabel(f"GT")
        plt.ylabel(f"{rfmp_name}")
        r_val, _ = pearsonr(avg_bot_gt_responses, avg_bot_rfmp_responses)
        plt.title(f"{layer_name}, r = {r_val:.4f}")
        
    pdf.savefig()
    plt.show()
    plt.close()
