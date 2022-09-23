"""
To visualize the difference between ground truth and bar mapping methods.

Tony Fu, Sep 20, 2022
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

is_random = False
this_is_a_test_run = False
map1_name = 'gt'                # ['gt', 'occlude']
map2_name = 'rfmp4a'            # ['rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']
fit_name = 'com'       # ['gaussian_fit', 'com', 'hot_spot']
fxvar_thres = 0.8


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
        top_x_df = top_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUX']]
        top_y_df = top_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUY']]
        bot_x_df = bot_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUX']]
        bot_y_df = bot_fit_df.loc[:, ['LAYER', 'UNIT', 'FXVAR', 'MUY']]
        
        # Rename columns from MUX -> TOP_X, etc.
        top_x_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'TOP_X']
        top_y_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'TOP_Y']
        bot_x_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'BOT_X']
        bot_y_df.columns = ['LAYER', 'UNIT', 'FXVAR', 'BOT_Y']
    else:
        fit_df = load_non_gaussian_fit_df(map_name, model_name, is_random, fit_name)
        fit_df = pad_missing_layers(fit_df)
        top_x_df = fit_df.loc[:, ['LAYER', 'UNIT', 'TOP_X']]
        top_y_df = fit_df.loc[:, ['LAYER', 'UNIT', 'TOP_Y']]
        bot_x_df = fit_df.loc[:, ['LAYER', 'UNIT', 'BOT_X']]
        bot_y_df = fit_df.loc[:, ['LAYER', 'UNIT', 'BOT_Y']]

    return top_x_df, top_y_df, bot_x_df, bot_y_df


def get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run):
    if this_is_a_test_run:  
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', f'{map1_name}_vs_{map2_name}', 'test')
    else:
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', f'{map1_name}_vs_{map2_name}', model_name)
    if not os.path.exists(result_dir):
        raise KeyError(f"{result_dir} does not exist.")
    return result_dir


# Pad the missing layers with NAN because not all layers are mapped.
gt_df = load_non_gaussian_fit_df('gt', model_name, is_random, 'com')
gt_no_data = gt_df[['LAYER', 'UNIT']].copy()  # template df used for padding
def pad_missing_layers(df):
    return pd.merge(gt_no_data, df, how='left')


def config_plot(limits):
    plt.plot([min(limits), max(limits)], [min(limits), max(limits)], 'k', alpha=0.4)
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')


###############################################################################

# Load the center of mass dataframes and pad the missing layers
top_x_df1, top_y_df1, bot_x_df1, bot_y_df1 = get_top_bot_xy_dfs(map1_name, model_name, is_random, fit_name)
top_x_df2, top_y_df2, bot_x_df2, bot_y_df2 = get_top_bot_xy_dfs(map2_name, model_name, is_random, fit_name)

result_dir = get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run)

pdf_path = os.path.join(result_dir, f"{model_name}_{map1_name}_{map2_name}_{fit_name}.pdf")
with PdfPages(pdf_path) as pdf:
    top_x_r_vals = []
    top_y_r_vals = []
    bot_x_r_vals = []
    bot_y_r_vals = []
    
    top_face_color = 'orange'
    bot_face_color = 'silver'
    
    # Filter data
    if fit_name == 'gaussian_fit':
        top_x_df1 = top_x_df1.loc[(top_x_df1.FXVAR > fxvar_thres) & (top_x_df2.FXVAR > fxvar_thres), :]
        top_x_df2 = top_x_df2.loc[(top_x_df1.FXVAR > fxvar_thres) & (top_x_df2.FXVAR > fxvar_thres), :]
        
        top_y_df1 = top_y_df1.loc[(top_y_df1.FXVAR > fxvar_thres) & (top_y_df2.FXVAR > fxvar_thres), :]
        top_y_df2 = top_y_df2.loc[(top_y_df1.FXVAR > fxvar_thres) & (top_y_df2.FXVAR > fxvar_thres), :]
        
        bot_x_df1 = bot_x_df1.loc[(bot_x_df1.FXVAR > fxvar_thres) & (bot_x_df2.FXVAR > fxvar_thres), :]
        bot_x_df2 = bot_x_df2.loc[(bot_x_df1.FXVAR > fxvar_thres) & (bot_x_df2.FXVAR > fxvar_thres), :]
        
        bot_y_df1 = bot_y_df1.loc[(bot_y_df1.FXVAR > fxvar_thres) & (bot_y_df2.FXVAR > fxvar_thres), :]
        bot_y_df2 = bot_y_df2.loc[(bot_y_df1.FXVAR > fxvar_thres) & (bot_y_df2.FXVAR > fxvar_thres), :]

    ###########################################################################
    #                   FIGURE 1. CORRELATIONS IN EACH LAYER                  #
    ###########################################################################
    plt.figure(figsize=(20, 9))
    if fit_name == 'gaussian_fit':
        plt.suptitle(f"{model_name}: {map1_name} vs {map2_name} ({fit_name}, fxvar_threshold = {fxvar_thres})", fontsize=24)
    else:
        plt.suptitle(f"{model_name}: {map1_name} vs {map2_name} ({fit_name})", fontsize=24)
    for conv_i, rf_size in enumerate(rf_sizes):
        # Skip Conv1
        if conv_i == 0:
            continue
        
        # Some basic layer info
        layer_name = f"conv{conv_i+1}"
        limits = (-75, 75)
        
        # Get data
        top_x1 = top_x_df1.loc[(top_x_df1.LAYER == layer_name), 'TOP_X']
        top_y1 = top_y_df1.loc[(top_y_df1.LAYER == layer_name), 'TOP_Y']
        bot_x1 = bot_x_df1.loc[(bot_x_df1.LAYER == layer_name), 'BOT_X']
        bot_y1 = bot_y_df1.loc[(bot_y_df1.LAYER == layer_name), 'BOT_Y']

        top_x2 = top_x_df2.loc[(top_x_df2.LAYER == layer_name), 'TOP_X']
        top_y2 = top_y_df2.loc[(top_y_df2.LAYER == layer_name), 'TOP_Y']
        bot_x2 = bot_x_df2.loc[(bot_x_df2.LAYER == layer_name), 'BOT_X']
        bot_y2 = bot_y_df2.loc[(bot_y_df2.LAYER == layer_name), 'BOT_Y']
        
        # Skip this layer if there is not data
        if len(top_x1) == 0 or len(top_x2) == 0:
            continue
        
        # Compute and record correlations
        top_x_r_vals.append(pearsonr(top_x1, top_x2)[0])
        top_y_r_vals.append(pearsonr(top_y1, top_y2)[0])
        bot_x_r_vals.append(pearsonr(bot_x1, bot_x2)[0])
        bot_y_r_vals.append(pearsonr(bot_y1, bot_y2)[0])
        
        # Plot x only
        plt.subplot(2,num_layers - 1, conv_i)
        plt.scatter(top_x1, top_x2, alpha=0.4, c='b')
        plt.text(-70,50,f'n = {len(top_x1)}\nr = {top_x_r_vals[-1]:.2f}', fontsize=16)
        config_plot(limits)
        plt.gca().set_facecolor(top_face_color)
        if conv_i == 1:
            plt.ylabel(f'{map2_name}', fontsize=16)
        plt.title(f"{layer_name}", fontsize=16)
        
        plt.subplot(2,num_layers - 1, num_layers+conv_i-1)
        plt.scatter(bot_x1, bot_x2, alpha=0.4, c='b')
        plt.text(-70,50,f'n = {len(bot_x1)}\nr = {bot_x_r_vals[-1]:.2f}', fontsize=16)
        config_plot(limits)
        plt.gca().set_facecolor(bot_face_color)
        plt.xlabel(f'{map1_name}', fontsize=16)
        if conv_i == 1:
            plt.ylabel(f'{map2_name}', fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()
    
    
    ###########################################################################
    #                     FIGURE 2. SUMMARY OF FIGURE 1.                      #
    ###########################################################################
    plt.figure(figsize=(12,5))
        
    x = np.arange(1,num_layers)
    x_str = [f"conv{num+1}" for num in x]
    plt.subplot(1,2,1)
    plt.plot(x, top_x_r_vals, 'b.-' ,markersize=20, label='x')
    plt.plot(x, top_y_r_vals, 'g.-' ,markersize=20, label='y')
    plt.ylabel('r', fontsize=16)
    plt.title("Top", fontsize=20)
    plt.xticks(x,x_str, fontsize=16)
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.1, 1.1)
    plt.gca().set_facecolor(top_face_color)
    plt.legend(fontsize=16)
    
    plt.subplot(1,2,2)
    plt.plot(x, bot_x_r_vals, 'b.-', markersize=20, label='x')
    plt.plot(x, bot_y_r_vals, 'g.-', markersize=20, label='y')
    plt.ylabel('r', fontsize=16)
    plt.title("Bottom", fontsize=20)
    plt.xticks(x,x_str, fontsize=16)
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.1, 1.1)
    plt.gca().set_facecolor(bot_face_color)
    plt.legend(fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()


    ###########################################################################
    #                FIGURE 3. ERROR DISTANCE IN EACH LAYER                   #
    ###########################################################################
    all_top_err_dist = np.array([])
    all_bot_err_dist = np.array([])
    
    plt.figure(figsize=(20, 9))
    if fit_name == 'gaussian_fit':
        plt.suptitle(f"{model_name}: {map1_name} vs {map2_name} ({fit_name}, fxvar_threshold = {fxvar_thres})", fontsize=24)
    else:
        plt.suptitle(f"{model_name}: {map1_name} vs {map2_name} ({fit_name})", fontsize=24)
    for conv_i, rf_size in enumerate(rf_sizes):
        # Skip Conv1
        if conv_i == 0:
            continue
        
        # Some basic layer info
        layer_name = f"conv{conv_i+1}"
        limits = (-75, 75)
        
        # Get data
        top_x1 = top_x_df1.loc[(top_x_df1.LAYER == layer_name), 'TOP_X']
        top_y1 = top_y_df1.loc[(top_y_df1.LAYER == layer_name), 'TOP_Y']
        bot_x1 = bot_x_df1.loc[(bot_x_df1.LAYER == layer_name), 'BOT_X']
        bot_y1 = bot_y_df1.loc[(bot_y_df1.LAYER == layer_name), 'BOT_Y']

        top_x2 = top_x_df2.loc[(top_x_df2.LAYER == layer_name), 'TOP_X']
        top_y2 = top_y_df2.loc[(top_y_df2.LAYER == layer_name), 'TOP_Y']
        bot_x2 = bot_x_df2.loc[(bot_x_df2.LAYER == layer_name), 'BOT_X']
        bot_y2 = bot_y_df2.loc[(bot_y_df2.LAYER == layer_name), 'BOT_Y']
        
        # Skip this layer if there is not data
        if len(top_x1) == 0 or len(top_x2) == 0:
            continue
        
        # Compute error distance
        top_err_dist = np.sqrt(np.square(top_x1 - top_x2) + np.square(top_y1 - top_y2))
        bot_err_dist = np.sqrt(np.square(bot_x1 - bot_x2) + np.square(bot_y1 - bot_y2))
        
        # Append to lists
        all_top_err_dist = np.append(all_top_err_dist, top_err_dist)
        all_bot_err_dist = np.append(all_bot_err_dist, bot_err_dist)

        # Plot the distribution of the error distance
        plt.subplot(2,num_layers - 1, conv_i)
        plt.hist(top_err_dist)
        plt.title(f"{layer_name} (n = {len(top_err_dist)})", fontsize=16)
        if conv_i == 1:
            plt.ylabel(f"count", fontsize=16)
        plt.gca().set_facecolor(top_face_color)
        
        plt.subplot(2,num_layers - 1, num_layers+conv_i-1)
        plt.hist(bot_err_dist)
        if conv_i == 1:
            plt.ylabel(f"count", fontsize=16)
        plt.gca().set_facecolor(bot_face_color)
        plt.xlabel(f"error distance (pix)", fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()

    
    ###########################################################################
    #           FIGURE 4. ERROR DISTANCE VS. COLOR ROTATION INDEX             #
    ###########################################################################
    # Save the error distances in dataframes:
    top_err_dist_df = top_x_df1.loc[(top_x_df1.LAYER != 'conv1'), ['LAYER', 'UNIT']]
    top_err_dist_df['ERR_DIST'] = all_top_err_dist

    bot_err_dist_df = bot_x_df1.loc[(bot_x_df1.LAYER != 'conv1'), ['LAYER', 'UNIT']]
    bot_err_dist_df['ERR_DIST'] = all_bot_err_dist
    
    # Load the color rotation index
    cri_path = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'cri', model_name, 'cri.txt')
    cri_df = pd.read_csv(cri_path, sep=" ", header=None)
    cri_df.columns = ['LAYER', 'UNIT', 'CRI']
    
    # Merge the two pds
    top_cri_err_dist_df = pd.merge(top_err_dist_df, cri_df, how='left')
    bot_cri_err_dist_df = pd.merge(bot_err_dist_df, cri_df, how='left')
    
    # Plot the relationship between error distance and CRI
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Relation between error distance ({map1_name} vs {map2_name}: {fit_name}) and CRI", fontsize=20)
    for conv_i, _ in enumerate(nums_units):
        if conv_i == 0:
            continue
        
        layer_name = f"conv{conv_i+1}"
        
        top_err_dist = top_cri_err_dist_df.loc[(top_cri_err_dist_df.LAYER == layer_name), 'ERR_DIST']
        top_cri      = top_cri_err_dist_df.loc[(top_cri_err_dist_df.LAYER == layer_name), 'CRI']
        bot_err_dist = bot_cri_err_dist_df.loc[(bot_cri_err_dist_df.LAYER == layer_name), 'ERR_DIST']
        bot_cri      = bot_cri_err_dist_df.loc[(bot_cri_err_dist_df.LAYER == layer_name), 'CRI']
        
        plt.subplot(2,num_layers - 1, conv_i)
        plt.scatter(top_err_dist, top_cri)
        rval, _ = pearsonr(top_err_dist, top_cri)
        if conv_i == 1:
            plt.ylabel("CRI", fontsize=16)
        plt.gca().set_facecolor(top_face_color)
        plt.text(45,1,f'n = {len(top_err_dist)}\nr = {rval:.2f}', fontsize=16)
        plt.title(layer_name, fontsize=16)
        plt.xlim([0, 70])
        plt.ylim([0, 2])
        
        
        plt.subplot(2,num_layers - 1, num_layers+conv_i-1)
        plt.scatter(bot_err_dist, bot_cri)
        rval, _ = pearsonr(bot_err_dist, bot_cri)
        plt.xlabel("error distance (pix)", fontsize=16)
        if conv_i == 1:
            plt.ylabel("CRI", fontsize=16)
        plt.gca().set_facecolor(bot_face_color)
        plt.text(45,1,f'n = {len(bot_err_dist)}\nr = {rval:.2f}', fontsize=16)
        plt.xlim([0, 70])
        plt.ylim([0, 2])
        
    pdf.savefig()
    plt.show()
    plt.close()
    

    ###########################################################################
    #             FIGURE 5. MAP CORRELATIONS VS. ERROR DISTANCE               #
    ###########################################################################
    max_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                 model_name, f"max_map_r.txt")
    min_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                 model_name, f"min_map_r.txt")
    max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)
    min_map_corr_df = pd.read_csv(min_map_corr_path, sep=" ", header=0)
    
    # Merge the two pds
    top_err_dist_map_corr_df = pd.merge(top_err_dist_df, max_map_corr_df, how='left')
    bot_err_dist_map_corr_df = pd.merge(bot_err_dist_df, min_map_corr_df, how='left')
    
    # Plot the relationship between error distance and map correlation
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Relation between error distance ({map1_name} vs {map2_name}: {fit_name}) and map correlation", fontsize=20)
    for conv_i, _ in enumerate(nums_units):
        if conv_i == 0:
            continue
        
        layer_name = f"conv{conv_i+1}"
        
        top_err_dist = top_err_dist_map_corr_df.loc[(top_err_dist_map_corr_df.LAYER == layer_name), 'ERR_DIST']
        top_map_corr = top_err_dist_map_corr_df.loc[(top_err_dist_map_corr_df.LAYER == layer_name), f"{map1_name}_vs_{map2_name}"]
        bot_err_dist = bot_err_dist_map_corr_df.loc[(bot_err_dist_map_corr_df.LAYER == layer_name), 'ERR_DIST']
        bot_map_corr = bot_err_dist_map_corr_df.loc[(bot_err_dist_map_corr_df.LAYER == layer_name), f"{map1_name}_vs_{map2_name}"]


        plt.subplot(2,num_layers - 1, conv_i)
        plt.scatter(top_err_dist, top_map_corr)
        rval, _ = pearsonr(top_err_dist, top_map_corr)
        if conv_i == 1:
            plt.ylabel("map correlation", fontsize=16)
        plt.gca().set_facecolor(top_face_color)
        plt.text(45,0.6,f'n = {len(top_err_dist)}\nr = {rval:.2f}', fontsize=16)
        plt.xlim([0, 70])
        plt.ylim([-1, 1])
        plt.title(layer_name, fontsize=16)
        
        plt.subplot(2,num_layers - 1, num_layers+conv_i-1)
        plt.scatter(bot_err_dist, bot_map_corr)
        rval, _ = pearsonr(bot_err_dist, bot_map_corr)
        plt.xlabel("error distance (pix)", fontsize=16)
        if conv_i == 1:
            plt.ylabel("map correlation", fontsize=16)
        plt.gca().set_facecolor(bot_face_color)
        plt.text(45,0.6,f'n = {len(bot_err_dist)}\nr = {rval:.2f}', fontsize=16)
        plt.xlim([0, 70])
        plt.ylim([-1, 1])

    pdf.savefig()
    plt.show()
    plt.close()
    
    
    ###########################################################################
    #                 FIGURE 6. MAP CORRELATIONS VS. CRI                      #
    ###########################################################################
    # Merge the two pds
    top_cri_map_corr_df = pd.merge(cri_df, max_map_corr_df, how='left')
    bot_cri_map_corr_df = pd.merge(cri_df, min_map_corr_df, how='left')

    # Plot the relationship between CRI and map correlation
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Relation between CRI and map correlation ({map1_name} vs {map2_name})", fontsize=20)
    for conv_i, _ in enumerate(nums_units):
        if conv_i == 0:
            continue
        
        layer_name = f"conv{conv_i+1}"
        
        top_cri = top_cri_map_corr_df.loc[(top_cri_map_corr_df.LAYER == layer_name), 'CRI']
        top_map_corr = top_cri_map_corr_df.loc[(top_cri_map_corr_df.LAYER == layer_name), f"{map1_name}_vs_{map2_name}"]
        bot_cri = bot_cri_map_corr_df.loc[(bot_cri_map_corr_df.LAYER == layer_name), 'CRI']
        bot_map_corr = bot_cri_map_corr_df.loc[(bot_cri_map_corr_df.LAYER == layer_name), f"{map1_name}_vs_{map2_name}"]

        plt.subplot(2,num_layers - 1, conv_i)
        plt.scatter(top_cri, top_map_corr)
        rval, _ = pearsonr(top_cri, top_map_corr)
        plt.title(layer_name, fontsize=16)
        if conv_i == 1:
            plt.ylabel("map correlation", fontsize=16)
        plt.gca().set_facecolor(top_face_color)
        plt.text(1,0.6,f'n = {len(top_cri)}\nr = {rval:.2f}', fontsize=16)
        plt.xlim([0, 2])
        plt.ylim([-1, 1])

        plt.subplot(2,num_layers - 1, num_layers+conv_i-1)
        plt.scatter(bot_cri, bot_map_corr)
        rval, _ = pearsonr(bot_cri, bot_map_corr)
        plt.xlabel("CRI", fontsize=16)
        if conv_i == 1:
            plt.ylabel("map correlation", fontsize=16)
        plt.gca().set_facecolor(bot_face_color)
        plt.text(1,0.6,f'n = {len(bot_cri)}\nr = {rval:.2f}', fontsize=16)
        plt.xlim([0, 2])
        plt.ylim([-1, 1])

    pdf.savefig()
    plt.show()
    plt.close()
