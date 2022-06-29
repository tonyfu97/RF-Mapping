"""
Script to generate gaussian fit images and statistics for the visualization
results.

Tony Fu, June 29, 2022
"""
import os
from PIL import Image

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path

from gaussian_fit import gaussian_fit
from hook import ConvUnitCounter


# Script guard.
if __name__ == "__main__":
    user_input = input("This code takes time to run. Are you sure? "\
                       "Enter 'y' to proceed. Type any other key to stop: ")
    if user_input == 'y':
        sum_mode = input("Choose a summation mode: {'sum', 'abs', 'sqr', 'relu'}: ")
        double_check = input(f"All .npy files in the result_dir will be deleted. Are you sure? (y/n): ")
        if user_input == 'y':
            pass
        else:
            raise KeyboardInterrupt("Interrupted by user")
    else: 
        raise KeyboardInterrupt("Interrupted by user")


model = models.alexnet(pretrained=True)
model_name = 'alexnet'
sum_mode = 'abs'
unit_counter = ConvUnitCounter(model)
layer_indicies, num_units = unit_counter.count()

backprop_sum_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/backprop_sum/{model_name}/{sum_mode}')

result_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/gaussian_fit/{model_name}')

for conv_i in range(len(layer_indicies)):
    
    for unit_i in range(num_units):
        
        max_backprop_sum_path = os.path.join(backprop_sum_dir, f"max_conv{conv_i+1}.{unit_i}.npy")
        min_backprop_sum_path = os.path.join(backprop_sum_dir, f"min_conv{conv_i+1}.{unit_i}.npy")
        max_backprop_sum = np.load(max_backprop_sum_path)
        min_backprop_sum = np.load(min_backprop_sum_path)
        
        
        
        
        



