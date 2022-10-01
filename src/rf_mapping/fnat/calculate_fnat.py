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
result_dir = os.path.join(c.REPO_DIR, 'results', 'fnat', rfmp_name, model_name)

# Txt file to save fnat
fnat_txt_path = os.path.join(result_dir, f"{rfmp_name}_fnat_{top_n_r}_avg.txt")
if os.path.exists(fnat_txt_path):
    os.remove(fnat_txt_path)

###############################################################################

# Script guard.
# if __name__ == "__main__":
#     user_input = input("This code takes time to run. Are you sure? "\
#                        "Enter 'y' to proceed. Type any other key to stop: ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")


# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)
num_layers = len(rf_sizes)


def config_plot(x, y):
    padding = 5
    rmin = min(x.min(), y.min()) - padding
    rmax = max(x.max(), y.max()) + padding

    plt.plot([rmin, rmax], [rmin, rmax], 'k', alpha=0.4)
    plt.xlim((rmin, rmax))
    plt.ylim((rmin, rmax))
    plt.gca().set_aspect('equal')


pdf_path = os.path.join(result_dir, f"{rfmp_name}_fnat_{top_n_r}_avg.pdf")
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
        top_rfmp_responses = top_rfmp_df.loc[(top_rfmp_df.RANK == 0), ['UNIT', 'R']]
        avg_top_rfmp_responses = top_rfmp_responses.groupby('UNIT').mean().to_numpy()
        avg_top_rfmp_responses = np.squeeze(avg_top_rfmp_responses, axis=1)
        
        bot_rfmp_responses = bot_rfmp_df.loc[(bot_rfmp_df.RANK == 0), ['UNIT', 'R']]
        avg_bot_rfmp_responses = bot_rfmp_responses.groupby('UNIT').mean().to_numpy()
        avg_bot_rfmp_responses = np.squeeze(avg_bot_rfmp_responses, axis=1)
        
        print(f"avg top rfmp has {np.sum(avg_top_rfmp_responses < 0):3} units less than zeros!")
        # print(f"avg bot rfmp has {np.sum(avg_bot_rfmp_responses < 0)} units less than zeros!")
        
        # Load the GT responses.
        gt_response_path = os.path.join(gt_response_dir, f"{layer_name}_responses.npy")
        gt_responses = np.load(gt_response_path)
        gt_responses = np.sort(gt_responses, axis=1)
        # Shape = [num_units, num_images, 2]. There are 2 columns:
        # 0. Max responses of the given image and unit
        # 1. Min responses of the given image and unit
        
        # Average the top- and bottom-N responses for each unit
        avg_top_gt_responses = np.mean(gt_responses[:, -top_n_r:, 0], axis=1)
        avg_bot_gt_responses = np.mean(gt_responses[:, :top_n_r, 1], axis=1)
        
        plt.subplot(2, num_layers, conv_i+1)
        plt.scatter(avg_top_gt_responses, avg_top_rfmp_responses)
        config_plot(avg_top_gt_responses, avg_top_rfmp_responses)
        plt.xlabel(f"GT")
        plt.ylabel(f"{rfmp_name}")
        r_val, _ = pearsonr(avg_top_gt_responses, avg_top_rfmp_responses)
        plt.title(f"{layer_name}, top, r = {r_val:.4f}")

        plt.subplot(2, num_layers, conv_i+1+num_layers)
        plt.scatter(avg_bot_gt_responses, avg_bot_rfmp_responses)
        config_plot(avg_bot_gt_responses, avg_bot_rfmp_responses)
        plt.xlabel(f"GT")
        plt.ylabel(f"{rfmp_name}")
        r_val, _ = pearsonr(avg_bot_gt_responses, avg_bot_rfmp_responses)
        plt.title(f"{layer_name}, bottom, r = {r_val:.4f}")
        
        # Compuate fnat
        top_fnat = avg_top_rfmp_responses / avg_top_gt_responses
        bot_fnat = avg_bot_rfmp_responses / avg_bot_gt_responses
        
        # Remove avg_gt_r that is less than 1 (or greater than -1 for bottom)
        top_fnat[avg_top_gt_responses < 1] = np.NaN
        top_fnat[avg_top_rfmp_responses < 1] = np.NaN
        bot_fnat[avg_bot_gt_responses > -1] = np.NaN
        bot_fnat[avg_bot_rfmp_responses > -1] = np.NaN
        
        # Save fnat in a txt file
        with open(fnat_txt_path, 'a') as f:
            for unit_i, (t, b) in enumerate(zip(top_fnat, bot_fnat)):
                f.write(f"{layer_name} {unit_i} {t:.4f} {b:.4f}\n")
        
    pdf.savefig()
    plt.show()
    plt.close()
