"""
We have two ways to quantify the similarity between two maps:
    1. Direct map correlations: flatten the two maps into linear arrays, then
                                compute the Pearson correlation coeff.
    2. Delta RF center: compute the RF center first using one of the 3 methods:
                        (gaussian_fit, hot spot peak, and center of mass), then
                        compute the Euclidean distance between the two centers.

This script helps to analyze how similar Methods 1 and 2 are by plotting the
scatter plots of Method 2 (of a particular layer) against Method 1. It then
plots a summary plot on the second page for all layers (except Conv1).

The hope is that this figure would argue for the use of 'hot spot peak' as an
alternative for Gaussian fit to find the RF centers, since the latter method
throws away too many units (e.g., only 105/192 remains in Alexnet Conv2 using a
fxvar threshold of 0.8).

Tony Fu, Sep 24, 2022
"""
import os
import sys
import math

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
from src.rf_mapping.result_txt_format import (GtGaussian as GTG,
                                              Rfmp4aWeighted as W,
                                              RfmpCOM as COM,
                                              RfmpHotSpot as HS)
from src.rf_mapping.hook import ConvUnitCounter

# Please specify the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

this_is_a_test_run = False
map1_name = 'gt'       # ['gt', 'occlude']
map2_name = 'rfmp4c7o'   # ['rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']
fxvar_thres = 0.8
conv_i_to_plot = 1


###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n]") 
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)
num_layers = len(rf_sizes)


###############################################################################


top_face_color = 'orange'
bot_face_color = 'silver'
is_random = False


#############################  HELPER FUNCTIONS  ##############################


def load_gaussian_fit_dfs(map_name, model_name, is_random):
    """
    Gaussian fit deserves its own loading function because the top and bottom
    fits are saved in separate txt files. For other fit methods like 'com' and
    'hotspot', please use load_non_gaussian_fit_df().
    """
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""
    
    if map_name == 'gt':
        top_df_path = os.path.join(mapping_dir,
                               'ground_truth',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               'abs',
                               f"{model_name}_{map_name}_gaussian_top.txt")
        bot_df_path = os.path.join(mapping_dir,
                               'ground_truth',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               'abs',
                               f"{model_name}_{map_name}_gaussian_bot.txt")
    elif map_name == 'occlude':
        top_df_path = os.path.join(mapping_dir,
                               'occlude',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               f"{model_name}_{map_name}_gaussian_top.txt")
        bot_df_path = os.path.join(mapping_dir,
                               'occlude',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               f"{model_name}_{map_name}_gaussian_bot.txt")
    elif map_name in ('rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu'):
        top_df_path = os.path.join(mapping_dir,
                               map_name,
                               'gaussian_fit',
                               model_name,
                               f"weighted_top.txt")
        bot_df_path = os.path.join(mapping_dir,
                               map_name,
                               'gaussian_fit',
                               model_name,
                               f"weighted_bot.txt")
    else:
        raise KeyError(f"{map_name} does not exist.")

    top_fit_df = pd.read_csv(top_df_path, sep=" ", header=None)
    bot_fit_df = pd.read_csv(bot_df_path, sep=" ", header=None)
    
    # Name the columns. 
    # Note: The gaussian.txt of GT data doesn't have the number of bars, so we
    #       cannot use the 'W-format' to name their columns.
    if map_name in ('gt', 'occlude'):
        top_fit_df.columns = [e.name for e in GTG]
        bot_fit_df.columns = [e.name for e in GTG]
    else:
        top_fit_df.columns = [e.name for e in W]
        bot_fit_df.columns = [e.name for e in W]

    return top_fit_df, bot_fit_df


def load_non_gaussian_fit_df(map_name, model_name, is_random, fit_name):
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""
    fit_format = {'com' : COM, 'hot_spot' : HS}
    
    if map_name == 'gt':
        df_path = os.path.join(mapping_dir,
                               'ground_truth',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               'abs',
                               f"{model_name}_{map_name}_{fit_name}.txt")
    elif map_name in ('occlude', 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu'):
        df_path = os.path.join(mapping_dir,
                               map_name,
                               'gaussian_fit',
                               model_name,
                               f"{model_name}_{map_name}_{fit_name}.txt")
    else:
        raise KeyError(f"{map_name} does not exist.")

    fit_df = pd.read_csv(df_path, sep=" ", header=None)
    fit_df.columns = [e.name for e in fit_format[fit_name]]  # Name the columns
    return fit_df


def get_top_bot_xy_dfs(map_name, model_name, is_random, fit_name):
    if fit_name == 'gaussian_fit':
        top_fit_df, bot_fit_df = load_gaussian_fit_dfs(map_name, model_name, is_random)
        top_fit_df = pad_missing_layers(top_fit_df)
        bot_fit_df = pad_missing_layers(bot_fit_df)
        top_xy_df = top_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUX', 'MUY']]
        bot_xy_df = bot_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUX', 'MUY']]
        
        # Rename columns from MUX -> TOP_X, etc.
        top_xy_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'TOP_X', 'TOP_Y']
        bot_xy_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'BOT_X', 'BOT_Y']
    else:
        fit_df = load_non_gaussian_fit_df(map_name, model_name, is_random, fit_name)
        fit_df = pad_missing_layers(fit_df)
        top_xy_df = fit_df.loc[:, ['LAYER', 'UNIT', 'TOP_X', 'TOP_Y']]
        bot_xy_df = fit_df.loc[:, ['LAYER', 'UNIT', 'BOT_X', 'BOT_Y']]

    return top_xy_df, bot_xy_df


def get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run):
    if this_is_a_test_run:  
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare',
                                  f"{map1_name}_vs_{map2_name}", 'test')
    else:
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare',
                                  f"{map1_name}_vs_{map2_name}", model_name)
    if not os.path.exists(result_dir):
        raise KeyError(f"{result_dir} does not exist.")
    return result_dir


# Pad the missing layers with NAN because not all layers are mapped.
gt_df = load_non_gaussian_fit_df('gt', model_name, is_random, 'com')
gt_no_data = gt_df[['LAYER', 'UNIT']].copy()  # template df used for padding
def pad_missing_layers(df):
    return pd.merge(gt_no_data, df, how='left')


################################# LOAD DATA ###################################


# Load the center of mass dataframes and pad the missing layers: Ground truth or occluder
top_xy_df_gaussian1, bot_xy_df_gaussian1 =\
            get_top_bot_xy_dfs(map1_name, model_name, is_random, 'gaussian_fit')
top_xy_df_hot_spot1, bot_xy_df_hot_spot1 =\
            get_top_bot_xy_dfs(map1_name, model_name, is_random, 'hot_spot')
top_xy_df_com1, bot_xy_df_com1 =\
            get_top_bot_xy_dfs(map1_name, model_name, is_random, 'com')


# Load the center of mass dataframes and pad the missing layers: Artificial stimuli
top_xy_df_gaussian2, bot_xy_df_gaussian2 =\
            get_top_bot_xy_dfs(map2_name, model_name, is_random, 'gaussian_fit')
top_xy_df_hot_spot2, bot_xy_df_hot_spot2 =\
            get_top_bot_xy_dfs(map2_name, model_name, is_random, 'hot_spot')
top_xy_df_com2, bot_xy_df_com2 =\
            get_top_bot_xy_dfs(map2_name, model_name, is_random, 'com')


# Load the correlation data
max_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                 model_name, f"max_map_r.txt")
min_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                 model_name, f"min_map_r.txt")
max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)
min_map_corr_df = pd.read_csv(min_map_corr_path, sep=" ", header=0)


#############  CALCULATE ERROR DISTANCES BETWEEN MAP1 and MAP2  ###############

# Using max/min_map_corr_df as a base dataframe, construct a dataframe we
# are going to store the error distances of each fit method.

max_df = max_map_corr_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']]
min_df = min_map_corr_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']]

max_df.loc[:,'GAUSSIAN_ERR_DIST'] = np.sqrt(
                                    np.square(top_xy_df_gaussian1.loc[:,'TOP_X'] - top_xy_df_gaussian2.loc[:,'TOP_X']) + 
                                    np.square(top_xy_df_gaussian1.loc[:,'TOP_Y'] - top_xy_df_gaussian2.loc[:,'TOP_Y']))
max_df.loc[:, 'GAUSSIAN_FXVAR_TOO_LOW'] = np.logical_or(top_xy_df_gaussian1.loc[:,'FXVAR'] < fxvar_thres, top_xy_df_gaussian2.loc[:,'FXVAR'] < fxvar_thres)
max_df.loc[:,'HOT_SPOT_ERR_DIST'] = np.sqrt(
                                    np.square(top_xy_df_hot_spot1.loc[:,'TOP_X'] - top_xy_df_hot_spot2.loc[:,'TOP_X']) + 
                                    np.square(top_xy_df_hot_spot1.loc[:,'TOP_Y'] - top_xy_df_hot_spot2.loc[:,'TOP_Y']))
max_df.loc[:,'COM_ERR_DIST'] = np.sqrt(
                                np.square(top_xy_df_com1.loc[:,'TOP_X'] - top_xy_df_com2.loc[:,'TOP_X']) + 
                                np.square(top_xy_df_com1.loc[:,'TOP_Y'] - top_xy_df_com2.loc[:,'TOP_Y']))


min_df.loc[:,'GAUSSIAN_ERR_DIST'] = np.sqrt(
                                    np.square(bot_xy_df_gaussian1.loc[:,'BOT_X'] - bot_xy_df_gaussian2.loc[:,'BOT_X']) + 
                                    np.square(bot_xy_df_gaussian1.loc[:,'BOT_Y'] - bot_xy_df_gaussian2.loc[:,'BOT_Y']))
min_df.loc[:, 'GAUSSIAN_FXVAR_TOO_LOW'] = np.logical_or(bot_xy_df_gaussian1.loc[:,'FXVAR'] < fxvar_thres, bot_xy_df_gaussian2.loc[:,'FXVAR'] < fxvar_thres)
min_df.loc[:,'HOT_SPOT_ERR_DIST'] = np.sqrt(
                                    np.square(bot_xy_df_hot_spot1.loc[:,'BOT_X'] - bot_xy_df_hot_spot2.loc[:,'BOT_X']) + 
                                    np.square(bot_xy_df_hot_spot1.loc[:,'BOT_Y'] - bot_xy_df_hot_spot2.loc[:,'BOT_Y']))
min_df.loc[:,'COM_ERR_DIST'] = np.sqrt(
                                np.square(bot_xy_df_com1.loc[:,'BOT_X'] - bot_xy_df_com2.loc[:,'BOT_X']) + 
                                np.square(bot_xy_df_com1.loc[:,'BOT_Y'] - bot_xy_df_com2.loc[:,'BOT_Y']))


############  COMPUTE CORRELATIONS BETWEEN MAP_CORR and ERR_DIST  #############

top_plot_xy = []
bot_plot_xy = []

top_gaussian_n_r = []
bot_gaussian_n_r = []
top_hot_spot_n_r = []
bot_hot_spot_n_r = []
top_com_n_r = []
bot_com_n_r = []

for conv_i, rf_size in enumerate(rf_sizes):
    
    # Some basic layer info
    layer_name = f"conv{conv_i+1}"
    
    # Compute r values
    top_map_corr = max_df.loc[(max_df.LAYER == layer_name) & ~(max_df.GAUSSIAN_FXVAR_TOO_LOW), f'{map1_name}_vs_{map2_name}']
    top_gaussian_err_dist = max_df.loc[(max_df.LAYER == layer_name) & ~(max_df.GAUSSIAN_FXVAR_TOO_LOW), 'GAUSSIAN_ERR_DIST']
    r_val, p_val = pearsonr(top_map_corr, top_gaussian_err_dist)
    top_gaussian_n_r.append((len(top_map_corr), r_val))
    if conv_i == conv_i_to_plot:  # Store the data to a list for the layer that will be plotted below as an example
        top_plot_xy.append((top_gaussian_err_dist, top_map_corr))
    
    bot_map_corr = min_df.loc[(min_df.LAYER == layer_name) & ~(min_df.GAUSSIAN_FXVAR_TOO_LOW), f'{map1_name}_vs_{map2_name}']
    bot_gaussian_err_dist = min_df.loc[(min_df.LAYER == layer_name) & ~(min_df.GAUSSIAN_FXVAR_TOO_LOW), 'GAUSSIAN_ERR_DIST']
    r_val, p_val = pearsonr(bot_map_corr, bot_gaussian_err_dist)
    bot_gaussian_n_r.append((len(bot_map_corr), r_val))
    if conv_i == conv_i_to_plot:
        bot_plot_xy.append((bot_gaussian_err_dist, bot_map_corr))
    
    top_map_corr = max_df.loc[max_df.LAYER == layer_name, f'{map1_name}_vs_{map2_name}']
    top_hot_spot_err_dist = max_df.loc[max_df.LAYER == layer_name, 'HOT_SPOT_ERR_DIST']
    r_val, p_val = pearsonr(top_map_corr, top_hot_spot_err_dist)
    top_hot_spot_n_r.append((len(top_map_corr), r_val))
    if conv_i == conv_i_to_plot:
        top_plot_xy.append((top_hot_spot_err_dist, top_map_corr))
    
    bot_map_corr = min_df.loc[min_df.LAYER == layer_name, f'{map1_name}_vs_{map2_name}']
    bot_hot_spot_err_dist = min_df.loc[min_df.LAYER == layer_name, 'HOT_SPOT_ERR_DIST']
    r_val, p_val = pearsonr(bot_map_corr, bot_hot_spot_err_dist)
    bot_hot_spot_n_r.append((len(bot_map_corr), r_val))
    if conv_i == conv_i_to_plot:
        bot_plot_xy.append((bot_hot_spot_err_dist, bot_map_corr))
    
    top_com_err_dist = max_df.loc[max_df.LAYER == layer_name, 'COM_ERR_DIST']
    r_val, p_val = pearsonr(top_map_corr, top_com_err_dist)
    top_com_n_r.append((len(top_map_corr), r_val))
    if conv_i == conv_i_to_plot:
        top_plot_xy.append((top_com_err_dist, top_map_corr))
    
    bot_com_err_dist = min_df.loc[min_df.LAYER == layer_name, 'COM_ERR_DIST']
    r_val, p_val = pearsonr(bot_map_corr, bot_com_err_dist)
    bot_com_n_r.append((len(bot_map_corr), r_val))
    if conv_i == conv_i_to_plot:
        bot_plot_xy.append((bot_com_err_dist, bot_map_corr))


###############################################################################

     
result_dir = get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run)
layer_name = f"conv{conv_i_to_plot+1}"
pdf_path = os.path.join(result_dir, f"{model_name}_{layer_name}_{map1_name}_{map2_name}_figure_fit_methods.pdf")
with PdfPages(pdf_path) as pdf:
    def config_plot(r, n, face_color):
        plt.xlim([-1, 50])
        plt.ylim([-1.1, 1.1])
        plt.yticks([-1, -0.5, 0, 0.5, 1])
        plt.text(5, -0.9, f"r = {r:.4f}\nn = {n}", fontsize=18)
        plt.gca().set_facecolor(face_color)
    
    plt.figure(figsize=(18, 12))
    if map1_name == 'gt':
        map1_alias = 'guided backprop'
    plt.suptitle(f"Comparing {map1_alias} and {map2_name} using different criteria\n({model_name} {layer_name})",
                 fontsize=22)
    
    plt.subplot(2,3,1)
    plt.scatter(*top_plot_xy[0], alpha=0.4)
    n, r = top_gaussian_n_r[conv_i_to_plot]
    config_plot(r, n, top_face_color)
    plt.ylabel('Direct map correlation\n(top maps)', fontsize=18)
    plt.title(f'Gaussian fit\n(fxvar threshold = {fxvar_thres})', fontsize=18)
    
    plt.subplot(2,3,2)
    plt.scatter(*top_plot_xy[1], alpha=0.4)
    n, r = top_hot_spot_n_r[conv_i_to_plot]
    config_plot(r, n, top_face_color)
    plt.title('Hot spot peak', fontsize=18)
    
    plt.subplot(2,3,3)
    plt.scatter(*top_plot_xy[2], alpha=0.4)
    n, r = top_com_n_r[conv_i_to_plot]
    config_plot(r, n, top_face_color)
    plt.title('Center of mass', fontsize=18)
    
    plt.subplot(2,3,4)
    plt.scatter(*bot_plot_xy[0], alpha=0.4)
    n, r = bot_gaussian_n_r[conv_i_to_plot]
    config_plot(r, n, bot_face_color)
    plt.xlabel('$\Delta$ RF center (pix)', fontsize=18)
    plt.ylabel('Direct map correlation\n(bottom maps)', fontsize=18)
    
    plt.subplot(2,3,5)
    plt.scatter(*bot_plot_xy[1], alpha=0.4)
    n, r = bot_hot_spot_n_r[conv_i_to_plot]
    # plt.xlabel('$\Delta$ RF center (pix)', fontsize=18)
    config_plot(r, n, bot_face_color)
    
    plt.subplot(2,3,6)
    plt.scatter(*bot_plot_xy[2], alpha=0.4)
    n, r = bot_com_n_r[conv_i_to_plot]
    # plt.xlabel('$\Delta$ RF center (pix)', fontsize=18)
    config_plot(r, n, bot_face_color)

    pdf.savefig()
    plt.show()
    plt.close()

    
    # Plot summary plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1,2,1)
    # There are alot of '[1:]' because we are skipping Conv1
    plt.plot([r for n, r in top_gaussian_n_r[1:]], 'r.-', markersize=18, label='Gaussian fit')
    plt.plot([r for n, r in top_hot_spot_n_r[1:]], 'c.-', markersize=18, label='Hot spot peak')
    plt.plot([r for n, r in top_com_n_r[1:]], 'g.-', markersize=18, label='Center of mass')
    plt.xticks([0, 1, 2, 3], [f'conv{i+1}' for i in range(1, num_layers)], fontsize=12)
    # plt.legend(fontsize=16)
    plt.ylabel("Correlation with\ndirect map correlation", fontsize=16)
    plt.ylim([-1, -0.2])
    plt.gca().set_facecolor(top_face_color)
    
    plt.subplot(1,2,2)
    plt.plot([r for n, r in bot_gaussian_n_r[1:]], 'r.-', markersize=18, label='Gaussian fit')
    plt.plot([r for n, r in bot_hot_spot_n_r[1:]], 'c.-', markersize=18, label='Hot spot peak')
    plt.plot([r for n, r in bot_com_n_r[1:]], 'g.-', markersize=18, label='Center of mass')
    plt.xticks([0, 1, 2, 3], [f'conv{i+1}' for i in range(1, num_layers)], fontsize=12)
    plt.legend(fontsize=16)
    # plt.ylabel("correlation with\ndirect map correlation", fontsize=16)
    plt.ylim([-1, -0.2])
    plt.gca().set_facecolor(bot_face_color)
    
    pdf.savefig()
    plt.show()
    plt.close()
    