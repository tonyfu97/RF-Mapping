"""
Script to correlate the maps and plot the maps of each units.

Tony Fu, August 21st, 2022
"""
import os
import sys
import math

import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.ndimage.filters import gaussian_filter
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


sys.path.append('../../..')
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = 'resnet18'
image_shape = (227, 227)
this_is_a_test_run = False
max_or_min = 'max'
font_size = 20
r_val_threshold = 0.7

# Result paths:
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR,
                              'results',
                              'compare',
                              'map_correlations',
                              'test')
else:
    result_dir = os.path.join(c.REPO_DIR,
                             'results',
                             'compare',
                             'map_correlations',
                              model_name)

###############################################################################

# Get some layer info:
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

# Define helper functions:
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
        return np.load(mapping_path)  # [unit, yn, xn]
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
        return np.transpose(maps, (0,2,3,1))  # Need the color channel for plots.
    elif map_name == 'rfmp4_sin1':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4_sin1',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_sinemaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.transpose(maps, (0,2,3,1))  # Need the color channel for plots.
    else:
        raise KeyError(f"{map_name} does not exist.")
    
def plot_r_val(r_val, p_val, font_size):
    plt.xticks([])
    plt.yticks([])
    plt.text(0.1, 0.7, f"r = {r_val:.4f}", fontsize=font_size, color='w')
    plt.text(0.1, 0.4, f"p = {p_val * 100:.4f}%", fontsize=font_size, color='w')
    if math.isfinite(r_val):
        ax = plt.gca()
        ax.set_facecolor((r_val/2 + 0.5, 1 - abs(r_val), 1-r_val/2 - 0.5))

# Make pdf:
for conv_i in range(num_layers):
    layer_name = f"conv{conv_i+1}"
    
    all_map_names = ['gt', 'gt_composite', 'occlude', 'rfmp4a', 'rfmp4c7o', 'rfmp_sin1']
    high_r_val_counts = {'gt_vs_gt_composite' : 0,
                         'gt_vs_occlude' : 0,
                         'gt_vs_rfmp4a' : 0,
                         'gt_vs_rfmp4c7o' : 0,
                         'gt_composite_vs_occlude' : 0,
                         'gt_composite_vs_rfmp4a' : 0,
                         'gt_composite_vs_rfmp4c7o' : 0,
                         'occlude_vs_rfmp4a' : 0,
                         'occlude_vs_rfmp4c7o' : 0,
                         'rfmp4a_vs_rfmp4c7o' : 0}
    
    pdf_path = os.path.join(result_dir, f"{layer_name}_{max_or_min}_map_r.pdf")
    with PdfPages(pdf_path) as pdf:
        try:
            gt_maps = load_maps('gt', layer_name, max_or_min)
            gt_max_maps = load_maps('gt', layer_name, 'max')
            gt_min_maps = load_maps('gt', layer_name, 'min')
            gt_max_min_maps = gt_max_maps + gt_min_maps
            occlude_maps = load_maps('occlude', layer_name, max_or_min)
            rfmp4a_maps = load_maps('rfmp4a', layer_name, max_or_min)
            rfmp4c7o_maps = load_maps('rfmp4c7o', layer_name, max_or_min)
            rfmp4c7o_maps = load_maps('rfmp4c7o', layer_name, max_or_min)
        except:
            break  # This layer was not mapped.

        num_units = gt_maps.shape[0]

        # Correlate the maps.
        for unit_i in tqdm(range(num_units)):
            """
            Subplot indices:

                              2.gt  3.gt(max+min) 4.occlude  5.rfmp4a  6.rfmp4c7o         
                7. gt          8        9            10         11         12
                13.gt(max+min)          15           16         17         18
                19.occlude                           22         23         24
                25.rfmp4a                                       29         30
                31.rfmp4c7o                                                36
            """
            # Smooth the maps with gaussian blur to get rid off local texture
            # that will influence direct correlation.
            sigma = occlude_maps[unit_i].shape[-1] / 30
            occlude_map = gaussian_filter(occlude_maps[unit_i], sigma=sigma)
            
            # Get the other maps of this unit
            gt_map = gt_maps[unit_i]
            gt_max_min_map = gt_max_min_maps[unit_i]
            rfmp4a_map = rfmp4a_maps[unit_i]
            rfmp4c7o_map = rfmp4c7o_maps[unit_i]
            
            all_maps = [gt_map, gt_max_min_map, occlude_map, rfmp4a_map, rfmp4c7o_map]
            
            # Normalize the maps
            for i in range(len(all_maps)):
                if all_maps[i].max() != 0:
                    all_maps[i] = all_maps[i] / all_maps[i].max()
            
            plt.figure(figsize=(4*len(all_maps), 4*len(all_maps) - 2))
            plt.suptitle(f"Correlations of different maps (no.{unit_i}, {max_or_min})", fontsize=32)
            
            # Plot the maps at the margin.
            for idx, map in enumerate(all_maps):
                plt.subplot(len(all_maps)+1, len(all_maps)+1, idx + 2)
                plt.imshow(map, cmap='gray')
                plt.title(all_map_names[idx], fontsize=font_size)
                
                plt.subplot(len(all_maps)+1, len(all_maps)+1,
                            (len(all_maps) + 1) * (idx + 1) + 1)
                plt.imshow(map, cmap='gray')
                plt.title(all_map_names[idx], fontsize=font_size)

            # Average the color channels of rfmp4c7o before correlation.
            rfmp4c7o_idx = all_map_names.index('rfmp4c7o')
            all_maps[rfmp4c7o_idx] = np.mean(all_maps[rfmp4c7o_idx], axis=2)

            # Display the r values and p values.
            for idx1, map1 in enumerate(all_maps):
                for idx2, map2 in enumerate(all_maps):
                    if idx1 <= idx2:
                        plt.subplot(len(all_maps)+1,
                                    len(all_maps)+1,
                                    (len(all_maps) + 1) * (idx1 + 1) + 2 + idx2)
                        r_val, p_val = pearsonr(map1.flatten(), map2.flatten())
                        plot_r_val(r_val, p_val, font_size)
                        if idx1 != idx2 and r_val > r_val_threshold:
                            name1 = all_map_names[idx1]
                            name2 = all_map_names[idx2]
                            high_r_val_counts[f'{name1}_vs_{name2}'] += 1

            pdf.savefig()
            plt.close()
        
        plt.figure(figsize=(25, 8))
        plt.bar(high_r_val_counts.keys(), high_r_val_counts.values())
        plt.ylabel('counts', fontsize=font_size)
        plt.title(f"Distribution of r values higher than {r_val_threshold}", fontsize=font_size)
        pdf.savefig()
        plt.show()
        plt.close()
