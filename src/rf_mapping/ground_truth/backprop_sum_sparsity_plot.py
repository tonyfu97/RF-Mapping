"""
Make a plot of how sparse (how many near zero values) are there in the maps.
Need this to show that Direct Correlation tends to overestimate the similarity
of deeper layers.

Tony Fu, May 3, 2023
"""
import os
import sys

import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.image import preprocess_img_for_plot
import src.rf_mapping.constants as c


# Please specify some details here:
# model = models.alexnet()
# model_name = "alexnet"
model = models.vgg16()
model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

# Please double-check the directories:
backprop_sum_dir = os.path.join(c.RESULTS_DIR, 'ground_truth',
                                'backprop_sum', model_name, 'abs')

pdf_dir = backprop_sum_dir
ATOL = 0.001

###############################################################################

# Get info of conv layers.
conv_counter = ConvUnitCounter(model)
layer_indices, nums_units = conv_counter.count()
num_layers = len(layer_indices)

max_sparsity_list = []
min_sparsity_list = []
for conv_i in range(num_layers):
    layer_name = f"conv{conv_i + 1}"
    num_units = nums_units[conv_i]
    
    # Load maps.
    max_sum_path = os.path.join(backprop_sum_dir, f"{layer_name}_max.npy")
    min_sum_path = os.path.join(backprop_sum_dir, f"{layer_name}_min.npy")
    max_sum = np.load(max_sum_path)
    min_sum = np.load(min_sum_path)
    
    # Get dimension
    u, y, x = max_sum.shape
    assert u == num_units
    num_pixels = y * x
    
    # Initialization
    max_sum_sparsity = np.zeros((num_units))
    min_sum_sparsity = np.zeros((num_units))
    
    for unit_i in tqdm(range(num_units)):
        max_sum_sparsity[unit_i] = np.sum(np.isclose(max_sum[unit_i], 0, atol=ATOL))/num_pixels
        min_sum_sparsity[unit_i] = np.sum(np.isclose(min_sum[unit_i], 0, atol=ATOL))/num_pixels

    max_sparsity_list.append(max_sum_sparsity)
    min_sparsity_list.append(min_sum_sparsity)


pdf_path = os.path.join(pdf_dir, f"sparsity.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(15,8))
    plt.suptitle(f"{model_name} sparsity (atol = {ATOL})", fontsize=20)
    
    
    plt.subplot(2,1,1)
    avg_max_sparsity = [s.mean() for s in max_sparsity_list]
    plt.plot(avg_max_sparsity, '.-', markersize=20)
    plt.boxplot(max_sparsity_list, positions=range(len(max_sparsity_list)), widths=0.6)
    plt.ylabel('Corr. coeff. (Facilitatory)', fontsize=16)
    plt.ylim([-0.1, 1])
    plt.xticks([])
    
    plt.subplot(2,1,2) 
    avg_min_sparsity = [s.mean() for s in min_sparsity_list]
    plt.plot(avg_min_sparsity, '.-', markersize=20)
    plt.boxplot(min_sparsity_list, positions=range(len(min_sparsity_list)), widths=0.6)
    plt.ylabel('Corr. coeff. (Suppressive)', fontsize=16)
    plt.ylim([-0.1, 1])
    plt.xticks(np.arange(num_layers), np.arange(1, num_layers+1), fontsize=16)
    plt.xlabel('conv_i', fontsize=20)

    pdf.savefig()
    plt.close
