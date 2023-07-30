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
rfmp_name = 'rfmp4c7o'
num_stim_list = [50, 100, 250, 500, 750, 1000, 1500, 2000, 5000]
fxvar_thres = 0.7
num_stim_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

source_dir = os.path.join(c.RESULTS_DIR, 'test_num_stim')
result_dir = source_dir

###############################################################################

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

top_face_color = 'orange'
bot_face_color = 'silver'


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


def load_corr_df(rfmp_name, model_name, layer_name, num_stim):
    corr_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                                str(num_stim), f"map_correlations.txt")

    # Load the txt files as pandas DF.
    corr_df  = pd.read_csv(corr_txt_path, sep=" ", header=None)

    # Name the columns
    corr_df.columns = ['LAYER', 'UNIT', 'TOP_CORR', 'BOT_CORR']

    return corr_df


def load_com_df(rfmp_name, model_name, layer_name, num_stim):
    com_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                                str(num_stim), f"com.txt")

    # Load the txt files as pandas DF.
    com_df  = pd.read_csv(com_txt_path, sep=" ", header=None)

    # Name the columns
    com_df.columns = ['LAYER', 'UNIT', 'TOP_ERR_DIST', 'BOT_ERR_DIST']

    return com_df


def load_hot_spot_df(rfmp_name, model_name, layer_name, num_stim):
    hot_spot_txt_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
                                str(num_stim), f"hot_spot.txt")

    # Load the txt files as pandas DF.
    hot_spot_df  = pd.read_csv(hot_spot_txt_path, sep=" ", header=None)

    # Name the columns
    hot_spot_df.columns = ['LAYER', 'UNIT', 'TOP_ERR_DIST', 'BOT_ERR_DIST']

    return hot_spot_df


def load_gt_gaussian_dfs(model_name):
    top_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit',
                            model_name, 'abs', f"{model_name}_gt_gaussian_top.txt")
    bot_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit',
                            model_name, 'abs', f"{model_name}_gt_gaussian_bot.txt")

    # Load the txt files as pandas DF.
    top_df = pd.read_csv(top_path, sep=" ", header=None)
    bot_df = pd.read_csv(bot_path, sep=" ", header=None)

    # Name the columns
    top_df.columns = [e.name for e in GTG]
    bot_df.columns = [e.name for e in GTG]

    return top_df, bot_df


def geo_mean(sd1, sd2):
    return np.sqrt(np.power(sd1, 2) + np.power(sd2, 2))


# Pad the missing layers with NAN because not all layers are mapped.
gt_df, _ = load_gt_gaussian_dfs(model_name)
gt_no_data = gt_df[['LAYER', 'UNIT']].copy()  # template df used for padding
def pad_missing_layers(df):
    return pd.merge(gt_no_data, df, how='left')


###############################################################################


top_avg_radius_dict = {}
top_num_units_dict = {}
top_avg_corr_dict = {}
top_avg_com_err_dist_dict = {}
top_avg_hot_spot_err_dist_dict = {}
top_avg_gaussian_fit_err_dist_dict = {}

bot_avg_radius_dict = {}
bot_num_units_dict = {}
bot_avg_corr_dict = {}
bot_avg_com_err_dist_dict = {}
bot_avg_hot_spot_err_dist_dict = {}
bot_avg_gaussian_fit_err_dist_dict = {}

for conv_i, rf_size in enumerate(rf_sizes):
    # skipping Conv1
    if conv_i == 0:
        continue

    layer_name = f"conv{conv_i + 1}"
    rf_size = rf_size[0]
    
    top_avg_radius_dict[conv_i] = []
    top_num_units_dict[conv_i] = []
    top_avg_corr_dict[conv_i] = []
    top_avg_com_err_dist_dict[conv_i] = []
    top_avg_hot_spot_err_dist_dict[conv_i] = []
    top_avg_gaussian_fit_err_dist_dict[conv_i] = []
    
    bot_avg_radius_dict[conv_i] = []
    bot_num_units_dict[conv_i] = []
    bot_avg_corr_dict[conv_i] = []
    bot_avg_com_err_dist_dict[conv_i] = []
    bot_avg_hot_spot_err_dist_dict[conv_i] = []
    bot_avg_gaussian_fit_err_dist_dict[conv_i] = []
    
    for num_stim in num_stim_list:
        # Load Gaussian fit results
        top_df, bot_df = load_gaussian_fit_df(rfmp_name, model_name, layer_name, num_stim)

        top_sd1data = top_df.loc[(top_df.LAYER == layer_name) & (top_df.FXVAR > fxvar_thres), 'SD1']
        top_sd2data = top_df.loc[(top_df.LAYER == layer_name) & (top_df.FXVAR > fxvar_thres), 'SD2']
        bot_sd1data = bot_df.loc[(bot_df.LAYER == layer_name) & (bot_df.FXVAR > fxvar_thres), 'SD1']
        bot_sd2data = bot_df.loc[(bot_df.LAYER == layer_name) & (bot_df.FXVAR > fxvar_thres), 'SD2']

        top_radii = geo_mean(top_sd1data, top_sd2data)
        # top_radii = top_radii[top_radii < rf_size/2]  # Remove radius that are too big (debatable?)
        bot_radii = geo_mean(bot_sd1data, bot_sd2data)
        # bot_radii = top_radii[bot_radii < rf_size/2]  # Remove radius that are too big (debatable?)
        
        top_avg_radius_dict[conv_i].append(np.mean(top_radii))
        top_num_units_dict[conv_i].append(len(top_radii))
        bot_avg_radius_dict[conv_i].append(np.mean(bot_radii))
        bot_num_units_dict[conv_i].append(len(bot_radii))
        
        # Load map correlations.
        corr_df = load_corr_df(rfmp_name, model_name, layer_name, num_stim)
        top_avg_corr_dict[conv_i].append(np.mean(corr_df['TOP_CORR']))
        bot_avg_corr_dict[conv_i].append(np.mean(corr_df['BOT_CORR']))
        
        # Load COM error distances
        com_df = load_com_df(rfmp_name, model_name, layer_name, num_stim)
        top_avg_com_err_dist_dict[conv_i].append(np.mean(com_df['TOP_ERR_DIST']))
        bot_avg_com_err_dist_dict[conv_i].append(np.mean(com_df['BOT_ERR_DIST']))
        
        # Load hot spot error distances.
        hot_spot_df = load_hot_spot_df(rfmp_name, model_name, layer_name, num_stim)
        top_avg_hot_spot_err_dist_dict[conv_i].append(np.mean(hot_spot_df['TOP_ERR_DIST']))
        bot_avg_hot_spot_err_dist_dict[conv_i].append(np.mean(hot_spot_df['BOT_ERR_DIST']))
        
        # Load GT coordinates
        gt_t_gaussian_df, gt_b_gaussian_df = load_gt_gaussian_dfs(model_name)

        # Extract mux and muy and filtering
        top_df = pad_missing_layers(top_df)
        gt_t_mux = gt_t_gaussian_df.loc[(gt_t_gaussian_df.LAYER == layer_name) & (gt_t_gaussian_df.FXVAR > fxvar_thres) & (top_df.FXVAR > fxvar_thres), 'MUX']
        gt_t_muy = gt_t_gaussian_df.loc[(gt_t_gaussian_df.LAYER == layer_name) & (gt_t_gaussian_df.FXVAR > fxvar_thres) & (top_df.FXVAR > fxvar_thres), 'MUY']
        t_mux = top_df.loc[(top_df.LAYER == layer_name) & (gt_t_gaussian_df.FXVAR > fxvar_thres) & (top_df.FXVAR > fxvar_thres), 'MUX']
        t_muy = top_df.loc[(top_df.LAYER == layer_name) & (gt_t_gaussian_df.FXVAR > fxvar_thres) & (top_df.FXVAR > fxvar_thres), 'MUY']
        
        bot_df = pad_missing_layers(bot_df)
        gt_b_mux = gt_b_gaussian_df.loc[(gt_b_gaussian_df.LAYER == layer_name) & (gt_b_gaussian_df.FXVAR > fxvar_thres) & (bot_df.FXVAR > fxvar_thres), 'MUX']
        gt_b_muy = gt_b_gaussian_df.loc[(gt_b_gaussian_df.LAYER == layer_name) & (gt_b_gaussian_df.FXVAR > fxvar_thres) & (bot_df.FXVAR > fxvar_thres), 'MUY']
        b_mux = top_df.loc[(bot_df.LAYER == layer_name) & (gt_b_gaussian_df.FXVAR > fxvar_thres) & (bot_df.FXVAR > fxvar_thres), 'MUX']
        b_muy = top_df.loc[(bot_df.LAYER == layer_name) & (gt_b_gaussian_df.FXVAR > fxvar_thres) & (bot_df.FXVAR > fxvar_thres), 'MUY']
        
        # and compute error distances.
        top_gaussian_fit_err_dists = np.sqrt(np.square(gt_t_mux - t_mux) + np.square(gt_t_muy - t_muy))
        top_avg_gaussian_fit_err_dist_dict[conv_i].append(np.mean(top_gaussian_fit_err_dists))
        bot_gaussian_fit_err_dists = np.sqrt(np.square(gt_b_mux - b_mux) + np.square(gt_b_muy - b_muy))
        bot_avg_gaussian_fit_err_dist_dict[conv_i].append(np.mean(bot_gaussian_fit_err_dists))


# pdf_path = os.path.join(result_dir, rfmp_name, model_name, layer_name,
#                         f"{rfmp_name}_{model_name}_{layer_name}_num_stim.pdf")
pdf_path = os.path.join(result_dir, rfmp_name, model_name,
                        f"{rfmp_name}_{model_name}_{layer_name}_r_thres.pdf")
with PdfPages(pdf_path) as pdf:
    for conv_i in range(num_layers):
        # skipping Conv1
        if conv_i == 0:
            continue

        plt.figure(figsize=(36, 6))
        plt.suptitle(f"{model_name} {layer_name} {rfmp_name} (top)", fontsize=20)
        
        plt.subplot(1,6,1)
        plt.plot(num_stim_list, top_avg_radius_dict[conv_i], '.-', markersize=20)
        plt.xlabel('number of stimuli included', fontsize=16)
        plt.ylabel('average radius (pix)', fontsize=16)
        
        plt.subplot(1,6,2)
        plt.plot(num_stim_list, top_num_units_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'number of units with fxvar above {fxvar_thres}', fontsize=16)
        plt.ylim([0, 384])
        
        plt.subplot(1,6,3)
        plt.plot(num_stim_list, top_avg_corr_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average direct correlation', fontsize=16)
        
        plt.subplot(1,6,4)
        plt.plot(num_stim_list, top_avg_com_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average COM error distance (pix)', fontsize=16)
        
        plt.subplot(1,6,5)
        plt.plot(num_stim_list, top_avg_hot_spot_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average hot spot error distance (pix)', fontsize=16)
        
        plt.subplot(1,6,6)
        plt.plot(num_stim_list, top_avg_gaussian_fit_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average gaussian fit error distance (pix)', fontsize=16)
        
        pdf.savefig()
        plt.show()
        plt.close()
        
        plt.figure(figsize=(36, 6))
        plt.suptitle(f"{model_name} {layer_name} {rfmp_name} (bottom)", fontsize=20)
        
        plt.subplot(1,6,1)
        plt.plot(num_stim_list, bot_avg_radius_dict[conv_i], '.-', markersize=20)
        plt.xlabel('number of stimuli included', fontsize=16)
        plt.ylabel('average radius (pix)', fontsize=16)
        
        plt.subplot(1,6,2)
        plt.plot(num_stim_list, bot_num_units_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'number of units with fxvar above {fxvar_thres}', fontsize=16)
        plt.ylim([0, 384])
        
        plt.subplot(1,6,3)
        plt.plot(num_stim_list, bot_avg_corr_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average direct correlation', fontsize=16)
        
        plt.subplot(1,6,4)
        plt.plot(num_stim_list, bot_avg_com_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average COM error distance (pix)', fontsize=16)
        
        plt.subplot(1,6,5)
        plt.plot(num_stim_list, bot_avg_hot_spot_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average hot spot error distance (pix)', fontsize=16)
        
        plt.subplot(1,6,6)
        plt.plot(num_stim_list, bot_avg_gaussian_fit_err_dist_dict[conv_i], '.-', markersize=20)
        plt.ylabel(f'average gaussian fit error distance (pix)', fontsize=16)
        
        pdf.savefig()
        plt.show()
        plt.close()


pdf_path = os.path.join(result_dir, rfmp_name, model_name,
                        f"{rfmp_name}_{model_name}_figure_r_thresholds.pdf")
with PdfPages(pdf_path) as pdf:
    
    fig, axes = plt.subplots(2, num_layers-1)
    fig.set_size_inches((num_layers-1)*7,12)
    fig.suptitle(f"Finding the optimal response threshold percentages for {model_name} {rfmp_name}",
                 fontsize=24)

    for conv_i in range(1, num_layers):
        # Top row
        ax1 = axes[0][conv_i-1]
        ax2 = axes[0][conv_i-1].twinx()
        ax1.plot(num_stim_list, top_avg_corr_dict[conv_i], 'r.-', markersize=20)
        ax2.plot(num_stim_list, top_avg_hot_spot_err_dist_dict[conv_i], 'b.-', markersize=20)
        
        ax1.set_ylim([0.2, 0.9])
        ax1.set_title(f"conv{conv_i+1}", fontsize=20)
        ax1.set_facecolor(top_face_color)
        ax1.tick_params(axis='y', colors='red')
        ax2.tick_params(axis='y', colors='blue')

        if conv_i == 1:
            ax1.set_ylabel('Direct map correlation\n(top)', color='r', fontsize=16)
            ax2.set_ylabel('Avg hot spot error distance (pix)', color='b', fontsize=16)
        else:
            ax1.set_yticks([])
        
        # Bottom row
        ax1 = axes[1][conv_i-1]
        ax2 = axes[1][conv_i-1].twinx()
        ax1.plot(num_stim_list, bot_avg_corr_dict[conv_i], 'r.-', markersize=20)
        ax2.plot(num_stim_list, bot_avg_hot_spot_err_dist_dict[conv_i], 'b.-', markersize=20)
        
        ax1.set_ylim([0.2, 0.9])
        ax1.set_facecolor(bot_face_color)
        ax1.tick_params(axis='y', colors='red')
        ax2.tick_params(axis='y', colors='blue')

        if conv_i == 1:
            ax1.set_xlabel('r threshold percentage', fontsize=16)
            ax1.set_ylabel('Direct map correlation\n(top)', color='r', fontsize=16)
            ax2.set_ylabel('Avg hot spot error distance (pix)', color='b', fontsize=16)
        else:
            ax1.set_yticks([])
    
    pdf.savefig()
    plt.show()
    plt.close()
