"""
Make a pdf of the sum of the gradient patches.

Tony Fu, June 30, 2022

Converted to matplotlib-efficient version on July 25. NOT TESTED YET.
"""
import os
import sys

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c


# Please specify some details here:
model_name = "alexnet"
num_layers = 5

# We don't have space for all units. Choose a few to plot only.
example_unit_ids = [0,1,2,3,4]
rf_sigma_ratio = 1/30

# Please double-check the directories:
guided_backprop_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth',
                                'backprop_sum', model_name, 'abs')
occlude_dir = os.path.join(c.REPO_DIR, 'results', 'occlude', 'mapping',
                           model_name)
output_pdf_path = os.path.join(c.REPO_DIR, 'results', 'figures', f"{model_name}_gt_examples.pdf")


with PdfPages(output_pdf_path) as pdf:
    plt.figure(figsize=(10,10))
    plt.suptitle(f"{model_name} Guided Backprop")
    plot_id = 1
    for i, conv_i in enumerate(range(num_layers)):
        layer_name = f"conv{conv_i + 1}"
        
        # Load maps.
        guided_backprop_path = os.path.join(guided_backprop_dir, f"{layer_name}_max.npy")
        guided_backprop_maps = np.load(guided_backprop_path)

        for j, unit_id in enumerate(example_unit_ids):
            plt.subplot(num_layers, len(example_unit_ids), plot_id)
            plot_id += 1
            
            plt.imshow(guided_backprop_maps[unit_id], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            
            if i == 0:
                plt.title(f"unit {unit_id}", fontsize=14)
            if j == 0:
                plt.ylabel(layer_name, fontsize=14)
            
    pdf.savefig()
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.suptitle(f"{model_name} Occluder smooth factor = {rf_sigma_ratio:.4f}")
    plot_id = 1
    for i, conv_i in enumerate(range(num_layers)):
        layer_name = f"conv{conv_i + 1}"
        
        # Load maps.
        occlude_path = os.path.join(occlude_dir, f"{layer_name}_max.npy")
        occlude_maps = np.load(occlude_path)
        
        _, rf_size, _ = occlude_maps.shape
        sigma = rf_sigma_ratio * rf_size

        for j, unit_id in enumerate(example_unit_ids):
            plt.subplot(num_layers, len(example_unit_ids), plot_id)
            plot_id += 1
            
            o_map = occlude_maps[unit_id]
            o_map = gaussian_filter(o_map, sigma=sigma)
            plt.imshow(o_map / o_map.max(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            
            if i == 0:
                plt.title(f"unit {unit_id}", fontsize=14)
            if j == 0:
                plt.ylabel(layer_name, fontsize=14)

    pdf.savefig()
    plt.show()
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.suptitle(f"{model_name} Guided backprop vs. Occlude")
    selected_units = [("conv2", 0),
                      ("conv2", 2),
                      ("conv3", 55),
                      ("conv3", 35)]
    num_units = len(selected_units)
    
    for i, (layer_name, unit_id) in enumerate(selected_units):
        # Load maps.
        guided_backprop_max_path = os.path.join(guided_backprop_dir, f"{layer_name}_max.npy")
        guided_backprop_max_maps = np.load(guided_backprop_max_path)
        guided_backprop_min_path = os.path.join(guided_backprop_dir, f"{layer_name}_min.npy")
        guided_backprop_min_maps = np.load(guided_backprop_min_path)
        
        occlude_path = os.path.join(occlude_dir, f"{layer_name}_max.npy")
        occlude_maps = np.load(occlude_path)
        _, rf_size, _ = occlude_maps.shape
        sigma = rf_sigma_ratio * rf_size
        o_map = occlude_maps[unit_id]
        o_map = gaussian_filter(o_map, sigma=sigma)

        plt.subplot(3, num_units, i + 1)
        plt.imshow(guided_backprop_max_maps[unit_id], cmap='gray')
        plt.title(f"{layer_name} unit {unit_id}")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel("Facilitatory\nGuided Backprop")
            
        
        plt.subplot(3, num_units, i + 1 + num_units)
        plt.imshow(guided_backprop_min_maps[unit_id], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel("Suppressive\nGuided Backprop")
        
        plt.subplot(3, num_units, i + 1 + 2 * num_units)
        plt.imshow(o_map / o_map.max(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel("Occluder")


    pdf.savefig()
    plt.show()
    plt.close()
