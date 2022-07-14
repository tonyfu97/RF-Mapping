"""
Receptive field mapping paradigm 4a.

Note: all code assumes that the y-axis points downward.

Tony Fu, July 13th, 2022
"""
import os
import sys

import numpy as np
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt

sys.path.append('..')
from mapping import BarRfMapperP4a
from spatial import get_rf_sizes
from files import delete_all_npy_files
import constants as c

# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = 'vgg16'
cumulate_modes = ['weighted', 'or']
xn = yn = 227
percent_max_min_to_cumulate = 0.1
this_is_a_test_run = False

# Please double-check the directories:
if this_is_a_test_run:
    result_dir = c.REPO_DIR + f'/results/rfmp4a/test/'
else:
    result_dir = c.REPO_DIR + f'/results/rfmp4a/{model_name}/'
pdf_dir = result_dir
grid_pdf_path = os.path.join(pdf_dir, f"grids.pdf")

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take time to run. Are you sure? ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Get info of conv layers.
layer_indices, rf_sizes = get_rf_sizes(model, (yn, xn))
num_layers = len(layer_indices)

delete_all_npy_files(result_dir)
for conv_i in range(num_layers):
    mapper = BarRfMapperP4a(model, conv_i, (yn, xn), percent_max_min_to_cumulate)

    if this_is_a_test_run:
        mapper.set_debug(True)

    mapper.map()
    mapper.save_maps(result_dir)

    for cumulate_mode in cumulate_modes:
        mapper.make_pdf(result_dir + f'conv{conv_i+1}_{cumulate_mode}_maps.pdf',
                            cumulate_mode)
