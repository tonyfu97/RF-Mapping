"""
Find the good number of bars used for our mapping methods.

Step 3. Plot the result and save it as an PDF.

Tony Fu, August 18th, 2022
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
from src.rf_mapping.gaussian_fit import theta_to_ori
from src.rf_mapping.gaussian_fit import GaussianFitParamFormat as ParamFormat
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import Rfmp4aWeighted as W


# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
image_shape = (227, 227)
this_is_a_test_run = False
batch_size = 10
conv_i_to_run = 1  # conv_i = 1 means Conv2
rfmp_name = 'rfmp4a'
num_stim_list = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]
fxvar_thres = 0.7

source_dir = os.path.join(c.REPO_DIR, 'results', 'test_num_stim')
result_dir = source_dir

###############################################################################

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)


# Helper functions.
def write_txt(f, layer_name, unit_i, raw_params, fxvar, map_size, num_bars):
    # Unpack params
    amp = raw_params[ParamFormat.A_IDX]
    mu_x = raw_params[ParamFormat.MU_X_IDX]
    mu_y = raw_params[ParamFormat.MU_Y_IDX]
    sigma_1 = raw_params[ParamFormat.SIGMA_1_IDX]
    sigma_2 = raw_params[ParamFormat.SIGMA_2_IDX]
    theta = raw_params[ParamFormat.THETA_IDX]
    offset = raw_params[ParamFormat.OFFSET_IDX]
    
    # Some primitive processings:
    # (1) move original from top-left to map center.s
    mu_x = mu_x - (map_size/2)
    mu_y = mu_y - (map_size/2)
    # (2) take the abs value of sigma values.
    sigma_1 = abs(sigma_1)
    sigma_2 = abs(sigma_2)
    # (3) convert theta to orientation.
    orientation = theta_to_ori(sigma_1, sigma_2, theta)

    f.write(f"{layer_name} {unit_i} ")
    f.write(f"{mu_x:.2f} {mu_y:.2f} ")
    f.write(f"{sigma_1:.2f} {sigma_2:.2f} ")
    f.write(f"{orientation:.2f} ")
    f.write(f"{amp:.3f} {offset:.3f} ")
    f.write(f"{fxvar:.4f} ")  # the fraction of variance explained by params
    f.write(f"{num_bars}\n")
    

def load_gaussian_fit_df(rfmp_name, model_name, layer_name, num_stim):
    top_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                                str(num_stim), f"gaussian_fit_weighted_top.txt")
    bot_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                                str(num_stim), f"gaussian_fit_weighted_bot.txt")

    # Load the txt files as pandas DF.
    top_df  = pd.read_csv(top_txt_path, sep=" ", header=None)
    bot_df  = pd.read_csv(bot_txt_path, sep=" ", header=None)

    # Name the columns
    top_df.columns = [e.name for e in W]
    bot_df.columns = [e.name for e in W]

    return top_df, bot_df


def geo_mean(sd1, sd2):
    return np.sqrt(np.power(sd1, 2) + np.power(sd2, 2))


layer_name = f"conv{conv_i_to_run + 1}"
rf_size = rf_sizes[conv_i_to_run][0]

pdf_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                        f"{rfmp_name}_{model_name}_{layer_name}_num_stim.pdf")
with PdfPages(pdf_path) as pdf:
    radius_list = []
    num_units_list = []
    for num_stim in num_stim_list:
        # Load Gaussian fit results
        top_df, bot_df = load_gaussian_fit_df(rfmp_name, model_name, layer_name, num_stim)
            
        sd1data = top_df.loc[(top_df.LAYER == layer_name) & (top_df.FXVAR > fxvar_thres), 'SD1']
        sd2data = top_df.loc[(top_df.LAYER == layer_name) & (top_df.FXVAR > fxvar_thres), 'SD2']
        
        radii = geo_mean(sd1data, sd2data)
        radii_without_too_big = radii[radii < rf_size]
        radius_list.append(np.mean(radii_without_too_big))
        num_units_list.append(len(radii_without_too_big))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1,2,1)
    plt.plot(num_stim_list, radius_list, '.-', markersize=20)
    plt.xlabel('number of stimuli included', fontsize=16)
    plt.ylabel('average radius (pix)', fontsize=16)
    
    plt.subplot(1,2,2)
    plt.plot(num_stim_list, num_units_list, '.-', markersize=20)
    plt.xlabel('number of stimuli included', fontsize=16)
    plt.ylabel(f'number of units with fxvar above {fxvar_thres}', fontsize=16)
    plt.ylim([0, 192])
    
    pdf.savefig()
    plt.show()
    plt.close()
