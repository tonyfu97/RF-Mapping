"""
To visualize the difference between ground truth and bar mapping methods.

Tony Fu, July 27th, 2022
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
from src.rf_mapping.result_txt_format import (RfmpHotSpot as HS)
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
map1_name = 'gt'
map2_name = 'rfmp4a'


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

def load_hot_spot_df(map_name, model_name, is_random):
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    is_random_str = "_random" if is_random else ""
    
    if map_name == 'gt':
        df_path = os.path.join(mapping_dir,
                               'ground_truth',
                               f'gaussian_fit{is_random_str}',
                               model_name,
                               'abs',
                               f"{model_name}_{map_name}_hot_spot.txt")
    elif map_name in ('occlude', 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu'):
        df_path = os.path.join(mapping_dir,
                               map_name,
                               'gaussian_fit',
                               model_name,
                               f"{model_name}_{map_name}_hot_spot.txt")
    else:
        raise KeyError(f"{map_name} does not exist.")

    hot_spot_df = pd.read_csv(df_path, sep=" ", header=None)
    
    # Name the columns:
    hot_spot_df.columns = [e.name for e in HS]
    
    return hot_spot_df


def get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run):
    if this_is_a_test_run:  
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', f'{map1_name}_vs_{map2_name}', 'test')
    else:
        result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', f'{map1_name}_vs_{map2_name}', model_name)
    if not os.path.exists(result_dir):
        raise KeyError(f"{result_dir} does not exist.")
    return result_dir


# Pad the missing layers with NAN because not all layers are mapped.
gt_df = load_hot_spot_df('gt', model_name, is_random)
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

# Load the hotspot dataframes and pad the missing layers
map1_df = load_hot_spot_df(map1_name, model_name, is_random)
map2_df = load_hot_spot_df(map2_name, model_name, is_random)
map1_df = pad_missing_layers(map1_df)
map2_df = pad_missing_layers(map2_df)

result_dir = get_result_dir(map1_name, map2_name, model_name, this_is_a_test_run)

pdf_path = os.path.join(result_dir, f"{model_name}_{map1_name}_and_{map2_name}_hot_spot.pdf")
with PdfPages(pdf_path) as pdf:
    top_x_r_vals = []
    top_y_r_vals = []
    bot_x_r_vals = []
    bot_y_r_vals = []
    
    top_face_color = 'orange'
    bot_face_color = 'silver'
    
    plt.figure(figsize=(20, 9))
    for conv_i, rf_size in enumerate(rf_sizes):
        # Skip Conv1
        if conv_i == 0:
            continue
        
        # Some basic layer info
        layer_name = f"conv{conv_i+1}"
        limits = (-75, 75)
        
        # Get data
        top_x1 = map1_df.loc[(map1_df.LAYER == layer_name), 'TOP_X']
        top_y1 = map1_df.loc[(map1_df.LAYER == layer_name), 'TOP_Y']
        bot_x1 = map1_df.loc[(map1_df.LAYER == layer_name), 'BOT_X']
        bot_y1 = map1_df.loc[(map1_df.LAYER == layer_name), 'BOT_Y']

        top_x2 = map2_df.loc[(map2_df.LAYER == layer_name), 'TOP_X']
        top_y2 = map2_df.loc[(map2_df.LAYER == layer_name), 'TOP_Y']
        bot_x2 = map2_df.loc[(map2_df.LAYER == layer_name), 'BOT_X']
        bot_y2 = map2_df.loc[(map2_df.LAYER == layer_name), 'BOT_Y']
        
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
