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

sys.path.append('../../..')
from src.rf_mapping.bar import rfmp4a_run_01b
import src.rf_mapping.constants as c

# Please specify some details here:
# model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = 'alexnet'
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
model_name = 'vgg16'  # Conv12 was not finished last time
this_is_a_test_run = False

# Please double-check the directories:
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', model_name)

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take time to run. Are you sure? [y/n] ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

rfmp4a_run_01b(model, model_name, result_dir, _debug=this_is_a_test_run)

"""
Tony - Below is the old way of running the rfmp4a. This old way uses the full
       image size 227 * 227. The new way above is much faster because it
       only uses the stimulus size that is just large enough to present the
       bar stimuli.

from mapping import BarRfMapperP4a
from spatial import get_rf_sizes
from files import delete_all_npy_files

xn = yn = 227
percent_max_min_to_cumulate = 0.1
cumulate_mode = ['weighted', 'or']

# Get info of conv layers.
layer_indices, rf_sizes = get_rf_sizes(model, (yn, xn))
num_layers = len(layer_indices)

for cumulate_mode in cumulate_modes:
    result_dir_with_mode = os.path.join(result_dir, cumulate_mode)
    delete_all_npy_files(result_dir_with_mode)

for conv_i in range(num_layers):
    if conv_i == 0: continue
    mapper = BarRfMapperP4a(model, conv_i, (yn, xn), percent_max_min_to_cumulate)
    mapper.set_debug(this_is_a_test_run)

    mapper.map()
    mapper.save_maps(result_dir)

    for cumulate_mode in cumulate_modes:
        result_dir_with_mode = os.path.join(result_dir, cumulate_mode, f'conv{conv_i+1}_maps.pdf')
        mapper.make_pdf(result_dir_with_mode, cumulate_mode)
"""