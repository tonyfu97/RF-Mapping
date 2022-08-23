"""
Script to correlate the maps and plot the maps of each units.

Tony Fu, August 21st, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append('../../..')
from src.rf_mapping.spatial import get_rf_sizes
import src.rf_mapping.constants as c


# Please specify some details here:
# model = models.alexnet(pretrained=True)
# model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
model = models.resnet18(pretrained=True)
model_name = 'resnet18'
image_shape = (227, 227)
this_is_a_test_run = False
max_or_min = 'min'
font_size = 20

# Result paths:
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
        return np.transpose(maps, (0,2,3,1))
    else:
        raise KeyError(f"{map_name} does not exist.")
    
def plot_r_val(r_val, p_val, font_size=20):
    plt.xticks([])
    plt.yticks([])
    plt.text(0.2, 0.7, f"r = {r_val:.4f}", fontsize=font_size)
    plt.text(0.1, 0.4, f"p = {p_val:.8f}", fontsize=font_size)
    
# Make pdf:
for conv_i in range(num_layers):
    layer_name = f"conv{conv_i+1}"
    pdf_path = os.path.join(result_dir, f"{layer_name}_{max_or_min}_map_r.pdf")
    with PdfPages(pdf_path) as pdf:
        try:
            gt_maps = load_maps('gt', layer_name, max_or_min)
            occlude_maps = load_maps('occlude', layer_name, max_or_min)
            rfmp4a_maps = load_maps('rfmp4a', layer_name, max_or_min)
            rfmp4c7o_maps = load_maps('rfmp4c7o', layer_name, max_or_min) 
        except:
            break

        num_units = gt_maps.shape[0]

        # Correlate the maps.
        for unit_i in range(num_units):
            """
                         gt  occlude  rfmp4a  rfmp4c7o         
                          2      3       4         5
                gt        7      8       9        10
                occlude   12    13      14        15
                rfmp4a    17    18      19        20
                rfmp4c7o  22    23      24        25
            """
            plt.figure(figsize=(20, 18))
            plt.suptitle(f"Correlations of different maps (no.{unit_i}, {max_or_min})", fontsize=32)

            plt.subplot(5, 5, 2)
            plt.imshow(gt_maps[unit_i]/gt_maps[unit_i].max(), cmap='gray')
            plt.title('gt', fontsize=font_size)
            
            plt.subplot(5, 5, 3)
            plt.imshow(occlude_maps[unit_i]/occlude_maps[unit_i].max(), cmap='gray')
            plt.title('occlude', fontsize=font_size)
            
            plt.subplot(5, 5, 4)
            plt.imshow(rfmp4a_maps[unit_i]/rfmp4a_maps[unit_i].max(), cmap='gray')
            plt.title('rfmp4a', fontsize=font_size)
            
            plt.subplot(5, 5, 5)
            plt.imshow(rfmp4c7o_maps[unit_i]/rfmp4c7o_maps[unit_i].max())
            plt.title('rfmp4c7o', fontsize=font_size)
            
            plt.subplot(5, 5, 6)
            plt.imshow(gt_maps[unit_i]/gt_maps[unit_i].max(), cmap='gray')
            plt.ylabel('gt', fontsize=font_size)
            
            plt.subplot(5, 5, 11)
            plt.imshow(occlude_maps[unit_i]/occlude_maps[unit_i].max(), cmap='gray')
            plt.ylabel('occlude', fontsize=font_size)
            
            plt.subplot(5, 5, 16)
            plt.imshow(rfmp4a_maps[unit_i]/rfmp4a_maps[unit_i].max(), cmap='gray')
            plt.ylabel('rfmp4a', fontsize=font_size)
            
            plt.subplot(5, 5, 21)
            plt.imshow(rfmp4c7o_maps[unit_i]/rfmp4c7o_maps[unit_i].max())
            plt.ylabel('rfmp4c7o', fontsize=font_size)
            
            # Just like RFMP4c7o, but the color channels are averaged.
            rfmp4c7o_array = np.mean(rfmp4c7o_maps, axis=3)[unit_i].flatten()
            
            plt.subplot(5, 5, 7)
            r_val, p_val = pearsonr(gt_maps[unit_i].flatten(), gt_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 8)
            r_val, p_val = pearsonr(gt_maps[unit_i].flatten(), occlude_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 9)
            r_val, p_val = pearsonr(gt_maps[unit_i].flatten(), rfmp4a_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 10)
            r_val, p_val = pearsonr(gt_maps[unit_i].flatten(), rfmp4c7o_array)
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 13)
            r_val, p_val = pearsonr(occlude_maps[unit_i].flatten(), occlude_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 14)
            r_val, p_val = pearsonr(occlude_maps[unit_i].flatten(), rfmp4a_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 15)
            r_val, p_val = pearsonr(occlude_maps[unit_i].flatten(), rfmp4c7o_array)
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 19)
            r_val, p_val = pearsonr(rfmp4a_maps[unit_i].flatten(), rfmp4a_maps[unit_i].flatten())
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 20)
            r_val, p_val = pearsonr(rfmp4a_maps[unit_i].flatten(), rfmp4c7o_array)
            plot_r_val(r_val, p_val, font_size)
            
            plt.subplot(5, 5, 25)
            r_val, p_val = pearsonr(rfmp4c7o_array, rfmp4c7o_array)
            plot_r_val(r_val, p_val, font_size)

            pdf.savefig()
            plt.close()
