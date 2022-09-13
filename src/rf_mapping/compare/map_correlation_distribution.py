"""
Script to correlate the maps and plot the distributions of r values.

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
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True)
# model_name = 'resnet18'
image_shape = (227, 227)
this_is_a_test_run = False
p_val_threshold = 0.05  # if greater than this, exclude from plot.

# Please specify the what maps to compare:
# Choices are: 'gt', 'occlude', 'rfmp4a', and 'rfmp4c7o'.
map1_name = 'rfmp4c7o'
map2_name = 'rfmp4c7o'

# Result paths:
result_pdf_path = os.path.join(c.REPO_DIR,
                               'results',
                               'compare',
                               'map_correlations',
                               model_name,
                               f"{map1_name}_vs_{map2_name}.pdf")
result_txt_path = os.path.join(c.REPO_DIR,
                               'results',
                               'compare',
                               'map_correlations',
                               model_name,
                               f"{map1_name}_vs_{map2_name}.txt")

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
        return np.mean(maps, axis=1)
    else:
        raise KeyError(f"{map_name} does not exist.")
        
# Delete previous files:
if os.path.exists(result_txt_path):
    os.remove(result_txt_path)
    
# Make pdf:
with PdfPages(result_pdf_path) as pdf:
    for map1_max_or_min in ['max', 'min']:
        for map2_max_or_min in ['max', 'min']:
            plt.figure(figsize=(num_layers*5, 5))
            plt.suptitle(f"Distribution of map correlations: {map1_name} ({map1_max_or_min}) and {map2_name} ({map2_max_or_min})", fontsize=16)
            for conv_i in range(num_layers):
                layer_name = f"conv{conv_i+1}"
                try:
                    maps1 = load_maps(map1_name, layer_name, map1_max_or_min)
                    maps2 = load_maps(map2_name, layer_name, map2_max_or_min)
                    # print(maps1.shape)
                    
                    # Controls (commented out)
                    # maps1 = np.random.rand(*maps1.shape)
                    # maps2 = np.random.rand(*maps1.shape)
                except:
                    break
                
                num_units = maps1.shape[0]
                r_vals = []
                p_vals = []
                
                # Correlate the maps.
                for unit_i in range(num_units):
                    r_val, p_val = pearsonr(maps1[unit_i].flatten(), maps2[unit_i].flatten())
                    r_vals.append(r_val)
                    p_vals.append(p_val)
                
                r_vals = np.array(r_vals)
                p_vals = np.array(p_vals)

                # Record r-vals and p-vals in text file.
                with open(result_txt_path, 'a') as f:
                    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
                        f.write(f"{layer_name} {i} {map1_max_or_min} {map2_max_or_min} {r:.4f} {p:.8f}\n")
                
                plt.subplot(1,num_layers,conv_i+1)
                bins = np.linspace(-1, 1, 40)
                data = r_vals[p_vals < p_val_threshold]
                plt.hist(data, bins=bins)
                plt.xlabel("r")
                plt.ylabel("counts")
                plt.xlim([-1, 1])
                plt.title(f"{layer_name} (n = {len(data)}/{len(p_vals)})")

            pdf.savefig()
            plt.close()
