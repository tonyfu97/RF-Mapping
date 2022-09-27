"""
Plot the number of bars used for each response threshold.

Tony Fu, August 25th, 2022
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.gaussian_fit import theta_to_ori
from src.rf_mapping.gaussian_fit import GaussianFitParamFormat as ParamFormat
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import (GtGaussian as GTG,
                                              Rfmp4aWeighted as W)


# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
image_shape = (227, 227)
this_is_a_test_run = False
r_thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

source_dir = os.path.join(c.REPO_DIR, 'results', 'test_num_stim')
result_dir = source_dir

###############################################################################

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

top_face_color = 'orange'
bot_face_color = 'silver'

# Initialize dictionaries used to store count data for each layer
all_top_4a_counts = {}
all_bot_4a_counts = {}
all_top_4c7o_counts = {}
all_bot_4c7o_counts = {}

# Load number of bars data
for conv_i in range(1, num_layers):
    layer_name = f"conv{conv_i+1}"
    
    # Initialize lists
    all_top_4a_counts[conv_i] = [[], []]
    all_bot_4a_counts[conv_i] = [[], []]
    all_top_4c7o_counts[conv_i] = [[], []]
    all_bot_4c7o_counts[conv_i] = [[], []]
    
    # Extract count data
    for r_thres in r_thres_list:
        rfmp4a_path = os.path.join(result_dir, 'rfmp4a', model_name, layer_name,
                                       str(r_thres), f"{model_name}_rfmp4a_weighted_counts.txt")
        rfmp4c7o_path = os.path.join(result_dir, 'rfmp4c7o', model_name, layer_name,
                                       str(r_thres), f"{model_name}_rfmp4c7o_weighted_counts.txt")

        rfmp4a_df = pd.read_csv(rfmp4a_path, sep=" ", header=None)
        rfmp4c7o_df = pd.read_csv(rfmp4c7o_path, sep=" ", header=None)
        
        rfmp4a_df.columns = ['LAYER', 'UNIT', 'TOP_COUNTS', 'BOT_COUNTS']
        rfmp4c7o_df.columns = ['LAYER', 'UNIT', 'TOP_COUNTS', 'BOT_COUNTS']
        
        num_units = len(rfmp4a_df['UNIT'].unique())
        
        top_4a_mean = np.mean(rfmp4a_df['TOP_COUNTS'])
        top_4a_std = np.std(rfmp4a_df['TOP_COUNTS'])
        all_top_4a_counts[conv_i][0].append(top_4a_mean)
        all_top_4a_counts[conv_i][1].append(top_4a_std  / math.sqrt(num_units))
        
        bot_4a_mean = np.mean(rfmp4a_df['BOT_COUNTS'])
        bot_4a_std = np.std(rfmp4a_df['BOT_COUNTS'])
        all_bot_4a_counts[conv_i][0].append(bot_4a_mean)
        all_bot_4a_counts[conv_i][1].append(bot_4a_std  / math.sqrt(num_units))
        
        top_4c7o_mean = np.mean(rfmp4c7o_df['TOP_COUNTS'])
        top_4c7o_std = np.std(rfmp4c7o_df['TOP_COUNTS'])
        all_top_4c7o_counts[conv_i][0].append(top_4c7o_mean)
        all_top_4c7o_counts[conv_i][1].append(top_4c7o_std  / math.sqrt(num_units))
        
        bot_4c7o_mean = np.mean(rfmp4c7o_df['BOT_COUNTS'])
        bot_4c7o_std = np.std(rfmp4c7o_df['BOT_COUNTS'])
        all_bot_4c7o_counts[conv_i][0].append(bot_4c7o_mean)
        all_bot_4c7o_counts[conv_i][1].append(bot_4c7o_std / math.sqrt(num_units))


# Now, make the pdf
pdf_path = os.path.join(result_dir, f"{model_name}_num_bars.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(5*(num_layers-1), 10))
    plt.suptitle(f"Number of bars included for different response thresholds: {model_name}",
                 fontsize=24)

    for conv_i in range(1, num_layers):
        layer_name = f"conv{conv_i+1}"

        plt.subplot(2, num_layers-1, conv_i)
        plt.yscale('log') #, nonposy='clip')
        plt.ylim([1, 120000])
        plt.grid(color='k', alpha=0.3, linestyle='-', linewidth=1, axis='y', which='both')
        plt.errorbar(r_thres_list, all_top_4a_counts[conv_i][0],
                    #  yerr = all_top_4a_counts[conv_i][1],
                    0,
                     fmt ='k-o', alpha=0.8, label='rfmp4a')
        plt.errorbar(r_thres_list, all_top_4c7o_counts[conv_i][0],
                    #  yerr = all_top_4c7o_counts[conv_i][1],
                     0, 
                     fmt ='c-o', alpha=0.8, label='rfmp4c7o')
        plt.title(f"{layer_name}", fontsize=18)
        plt.gca().set_facecolor(top_face_color)
        if conv_i == 1:
            plt.ylabel('avg number of bars included\n(top)', fontsize=14)
            plt.legend(fontsize=18)
        
        plt.subplot(2, num_layers-1, conv_i + num_layers - 1)
        plt.yscale('log') #, nonposy='clip')
        plt.ylim([1, 120000])
        plt.grid(color='k', alpha=0.3, linestyle='-', linewidth=1, axis='y', which='both')
        plt.errorbar(r_thres_list, all_bot_4a_counts[conv_i][0],
                    #  yerr = all_bot_4a_counts[conv_i][1],
                    0,
                     fmt ='k-o', alpha=0.8, label='rfmp4a')
        plt.errorbar(r_thres_list, all_bot_4c7o_counts[conv_i][0],
                    #  yerr = all_bot_4c7o_counts[conv_i][1],
                    0,
                     fmt ='c-o', alpha=0.8, label='rfmp4c7o')
        plt.gca().set_facecolor(bot_face_color)
        if conv_i == 1:
            plt.xlabel('r threshold percentage', fontsize=14)
            plt.ylabel('avg number of bars included\n(bottom)', fontsize=14)
            plt.legend(fontsize=18)
        
    pdf.savefig() 
    plt.show()
    plt.close()

