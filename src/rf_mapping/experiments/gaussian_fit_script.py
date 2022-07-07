"""
Script to generate gaussian fit pdf and statistics for the visualization
results.

Tony Fu, June 29, 2022
"""
import os
import sys

from torchvision import models

sys.path.append('..')
from gaussian_fit import make_pdf
from hook import ConvUnitCounter
import constants as c


# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
sum_modes = ['abs', 'sqr']
this_is_a_test_run = False

# Please double-check the directories:
backprop_sum_dir = c.REPO_DIR + f'/results/ground_truth/backprop_sum/{model_name}'
result_dir = c.REPO_DIR + f'/results/ground_truth/gaussian_fit/{model_name}'

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input(
        "This code may take time to run. Are you sure? "\
        f"All .npy files in {result_dir} will be deleted. (y/n): ") 
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()

for sum_mode in sum_modes:
    backprop_sum_dir_with_mode = os.path.join(backprop_sum_dir, sum_mode)
    result_dir_with_mode = os.path.join(result_dir, sum_mode)
    
    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"
        num_units = nums_units[conv_i]
        
        # Do only the first 5 unit during testing phase
        if this_is_a_test_run:
            num_units = 5
        
        best_file_names = [f"max_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
        worst_file_names = [f"min_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
        both_file_names = [f"both_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
        pdf_name = f"{layer_name}.gaussian.pdf"
        plot_title = f"{model_name} {layer_name} (sum mode = {sum_mode})"
        
        make_pdf(backprop_sum_dir_with_mode, best_file_names, worst_file_names,
                 both_file_names, result_dir_with_mode, pdf_name, plot_title)
