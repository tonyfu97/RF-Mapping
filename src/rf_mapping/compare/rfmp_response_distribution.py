"""
Code to plot the distribution of the RFMP statistics.

Tony Fu, Sep, 2022
"""
from genericpath import isfile
import os
import sys
import math

import numpy as np
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.spatial import get_num_layers
from src.rf_mapping.result_txt_format import CenterReponses as CR


# Please confirm the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

# Please specify the what maps to compare:
# Choices are: 'gt', 'occlude', 'rfmp4a', and 'rfmp4c7o'.
gt_method = 'gt'
ephys_method = 'rfmp4a'

# Please confirm the directories
source_dir = os.path.join(c.REPO_DIR, 'results', ephys_method, 'mapping', model_name)
result_dir = os.path.join(c.REPO_DIR, 'results', ephys_method, 'analysis', model_name)

# Result paths:
def load_maps(map_name, layer_name, max_or_min):
    """Loads the maps of the layer."""
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    
    if map_name == 'gt':
        mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    'backprop_sum',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_{max_or_min}.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'occlude':
        mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_{max_or_min}.npy")
        maps = np.load(mapping_path)  # [top_n, unit, y, x]
        return np.mean(maps, axis=0)
    elif map_name == 'rfmp4a':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4a',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'rfmp4c7o':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4c7o',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.mean(maps, axis=1)
    else:
        raise KeyError(f"{map_name} does not exist.")


# Load the statistics of the top and bottom 5000 bars.
num_layers = get_num_layers(model)

# Define helper functions
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]

def normalize_responses(responses):
    if not isinstance(responses, np.ndarray):
        responses = np.array(responses)
    r_max = responses.max()
    r_min = responses.min()
    r_range = r_max - r_min
    if r_range != 0:
        return (responses - r_min) / r_range
    else:
        return responses - r_min


for conv_i in range(num_layers):
    layer_name = f"conv{conv_i + 1}"
    top_file_path = os.path.join(source_dir, f'{layer_name}_top5000_responses.txt')
    bot_file_path = os.path.join(source_dir, f'{layer_name}_bot5000_responses.txt')
    top_r_df = pd.read_csv(top_file_path, sep=' ', header=None)
    bot_r_df = pd.read_csv(bot_file_path, sep=' ', header=None)

    # Name the columns.
    set_column_names(top_r_df, CR)
    set_column_names(bot_r_df, CR)
    
    num_units = len(top_r_df.UNIT.unique())
    
    gt_t_maps = load_maps(gt_method, layer_name, 'max')
    gt_b_maps = load_maps(gt_method, layer_name, 'min')
    rfmp_t_maps = load_maps(ephys_method, layer_name, 'max')
    rfmp_b_maps = load_maps(ephys_method, layer_name, 'min')

    top_r_vals = []
    bot_r_vals = []
    top_areas_under_curve = []
    bot_areas_under_curve = []
    top_r_max = []
    bot_r_max = []
    
    pdf_path = os.path.join(result_dir, f"{model_name}_{layer_name}_{gt_method}_and_{ephys_method}_responses.pdf")
    with PdfPages(pdf_path) as pdf:
        for unit_i in tqdm(range(num_units)):
            # blur the maps
            sigma = 2
            gt_t_map = gaussian_filter(gt_t_maps[unit_i], sigma=sigma)
            gt_b_map = gaussian_filter(gt_b_maps[unit_i], sigma=sigma)
            rfmp_t_map = gaussian_filter(rfmp_t_maps[unit_i], sigma=sigma)
            rfmp_b_map = gaussian_filter(rfmp_b_maps[unit_i], sigma=sigma)
            
            top_r_val, _ = pearsonr(gt_t_map.flatten(), rfmp_t_map.flatten())
            bot_r_val, _ = pearsonr(gt_b_map.flatten(), rfmp_b_map.flatten())

            top_responses = top_r_df.loc[(top_r_df.UNIT == unit_i), 'R']
            norm_top_responses = normalize_responses(top_responses)
            top_intergral = np.sum(norm_top_responses)
            if math.isfinite(top_r_val):
                top_r_vals.append(top_r_val)
                top_r_max.append(top_responses.tolist()[0])
                top_areas_under_curve.append(top_intergral)

            bot_responses = bot_r_df.loc[(bot_r_df.UNIT == unit_i), 'R']
            norm_bot_responses = normalize_responses(bot_responses)
            bot_intergral = np.sum(norm_bot_responses)
            if math.isfinite(bot_r_val):
                bot_r_vals.append(bot_r_val)
                bot_r_max.append(bot_responses.tolist()[0])
                bot_areas_under_curve.append(bot_intergral) 

            plt.figure(figsize=(10,6))
            plt.suptitle(f"{layer_name} no.{unit_i} 5000 responses")
            
            plt.subplot(2,3,1)
            plt.plot(np.arange(1, 5001), top_responses)
            plt.title(f"top (nauc = {top_intergral:.0f})")
            plt.ylabel('response')
            
            plt.subplot(2,3,2)
            plt.imshow(gt_t_map, cmap='gray')
            plt.title(f"{gt_method}, (r = {top_r_val:.2f})")
            
            plt.subplot(2,3,3)
            plt.imshow(rfmp_t_map, cmap='gray')
            plt.title(f"{ephys_method}")
            
            plt.subplot(2,3,4)
            plt.plot(np.arange(1, 5001), bot_responses)
            plt.title(f"bottom (nauc = {bot_intergral:.0f})")
            plt.xlabel('ranking')
            plt.ylabel('response')
            
            plt.subplot(2,3,5)
            plt.imshow(gt_b_map, cmap='gray')
            plt.title(f"{gt_method}, (r = {bot_r_val:.2f})")
            
            plt.subplot(2,3,6)
            plt.imshow(rfmp_b_map, cmap='gray')
            plt.title(f"{ephys_method}")

            pdf.savefig()
            plt.close()
            
        # Turn the lists into numpy arrays
        top_r_vals = np.array(top_r_vals)
        bot_r_vals = np.array(bot_r_vals)
        top_areas_under_curve = np.array(top_areas_under_curve)
        bot_areas_under_curve = np.array(bot_areas_under_curve)
        top_r_max = np.array(top_r_max)
        bot_r_max = np.array(bot_r_max)

        # Transform the r values
        top_r_vals = np.exp(top_r_vals)
        bot_r_vals = np.exp(bot_r_vals)
        
        total_top_r_val, _ = pearsonr(top_r_vals, top_areas_under_curve)
        total_bot_r_val, _ = pearsonr(bot_r_vals, bot_areas_under_curve)
        plt.figure(figsize=(10,5))
        plt.suptitle(f"{layer_name}")

        plt.subplot(1,2,1)
        plt.scatter(top_r_vals, top_areas_under_curve)
        plt.xlabel('r (gt and rfmp4a correlation)')
        plt.ylabel('area under sorted response curve')
        plt.title(f"top (r = {total_top_r_val:.2f})")
        
        plt.subplot(1,2,2)
        plt.scatter(bot_r_vals, bot_areas_under_curve)
        plt.xlabel('r (gt and rfmp4a correlation)')
        plt.ylabel('area under sorted response curve')
        plt.title(f"bottom (r = {total_bot_r_val:.2f})")

        pdf.savefig()
        plt.close()

        total_top_r_val, _ = pearsonr(top_r_vals, top_r_max)
        total_bot_r_val, _ = pearsonr(bot_r_vals, bot_r_max)
        plt.figure(figsize=(10,5))
        plt.suptitle(f"{layer_name}")

        plt.subplot(1,2,1)
        plt.scatter(top_r_vals, top_r_max)
        plt.xlabel('r (gt and rfmp4a correlation)')
        plt.ylabel('maximum response')
        plt.title(f"top (r = {total_top_r_val:.2f})")
        
        plt.subplot(1,2,2)
        plt.scatter(bot_r_vals, bot_r_max)
        plt.xlabel('r (gt and rfmp4a correlation)')
        plt.ylabel('maximum response')
        plt.title(f"bottom (r = {total_bot_r_val:.2f})")

        pdf.savefig()
        plt.close()
