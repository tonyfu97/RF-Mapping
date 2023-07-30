"""
Receptive field mapping paradigm z.

Note: all code assumes that the y-axis points downward.

Tony Fu, July 4th, 2022
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from spatial import (get_conv_output_shapes,
                     calculate_center,
                     get_rf_sizes,
                     RfGrid,
                     SpatialIndexConverter,)
from image import make_box, preprocess_img_to_tensor
from hook import ConvUnitCounter
from stimulus import draw_bar
from files import delete_all_npy_files
import constants as c

# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = 'vgg16'
xn = yn = 227
rf_blen_ratios = [3/4, 3/8, 3/16, 3/32]
rf_blen_ratio_strs = ['3/4', '3/8', '3/16', '3/32']
aspect_ratios = [1/2, 1/5, 1/10]
thetas = np.arange(0, 180, 22.5)
laa = 0.5
fgval = 1.0
bgval = 0.5
threshold = 1  # for threshold cumulation maps.
this_is_a_test_run = True

# Please double-check the directories:
if this_is_a_test_run:
    result_dir = c.RESULTS_DIR + f'/rfmpz/test/'
else:
    result_dir = c.RESULTS_DIR + f'/rfmpz/{model_name}/'
pdf_dir = result_dir
grid_pdf_path = os.path.join(pdf_dir, f"grids.pdf")

###############################################################################

# Initiate helper objects.
bar_locations = RfGrid(model, (yn, xn))
converter = SpatialIndexConverter(model, (yn, xn))
unit_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, rf_sizes = get_rf_sizes(model, (yn, xn))
layer_indices, nums_units = unit_counter.count()
conv_output_shapes = get_conv_output_shapes(model, (yn, xn))
num_layers = len(layer_indices)

# Define some script-specific helper functions.
def box_to_center(box):
    """Find the center index of the box."""
    y_min, x_min, y_max, x_max = box
    xc = (x_min + x_max)//2
    yc = (y_min + y_max)//2
    return xc, yc

# with PdfPages(grid_pdf_path) as pdf:
#     for i, rf_blen_ratio in enumerate(rf_blen_ratios):
#         for aspect_ratio in aspect_ratios:
#             plt.figure(figsize=(4*num_layers, 5))
#             plt.suptitle(f"Bar Length = {rf_blen_ratio_strs[i]} M, aspect_ratio = {aspect_ratio}", fontsize=24)
#             for conv_i, layer_index in enumerate(layer_indices):
#                 # Get spatial center of box.
#                 spatial_index = np.array(conv_output_shapes[conv_i][-2:])
#                 spatial_index = calculate_center(spatial_index)
#                 box = converter.convert(spatial_index, layer_index, 0, is_forward=False)
#                 xc, yc = box_to_center(box)

#                 # Create bar.
#                 rf_size = rf_sizes[conv_i][0]
#                 blen = round(rf_blen_ratio * rf_size)
#                 bwid = round(aspect_ratio * blen)
#                 grid_spacing = blen/2
#                 bar = draw_bar(xn, yn, xc, yc, 30, blen, bwid, laa, fgval, bgval)
                
#                 # Get grid coordinates.
#                 grid_coords = bar_locations.get_grid_coords(layer_index, spatial_index, grid_spacing)
#                 grid_coords_np = np.array(grid_coords)

#                 plt.subplot(1, num_layers, conv_i+1)
#                 plt.imshow(bar, cmap='gray', vmin=0, vmax=1)
#                 plt.title(f"conv{conv_i + 1}", fontsize=20)
#                 plt.plot(grid_coords_np[:, 0], grid_coords_np[:, 1], 'k.')

#                 boundary = 10
#                 plt.xlim([box[1] - boundary, box[3] + boundary])
#                 plt.ylim([box[0] - boundary, box[2] + boundary])

#                 rect = make_box(box, linewidth=2)
#                 ax = plt.gca()
#                 ax.add_patch(rect)
#                 ax.invert_yaxis()

#             pdf.savefig()
#             plt.show()
#             plt.close()


###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take time to run. Are you sure? ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

def truncated_model(x, model, layer_index):
    """
    Returns the output of the specified layer without forward passing to the
    subsequent layers.

    Parameters
    ----------
    x : torch.tensor
        The input. Should have dimension (1, 3, 2xx, 2xx).
    model : torchvision.model.Module
        The neural network (or the layer if in a recursive case).
    layer_index : int
        The index of the layer, the output of which will be returned. The
        indexing excludes container layers.

    Returns
    -------
    y : torch.tensor
        The output of layer with the layer_index.
    layer_index : int
        Used for recursive cases. Should be ignored.
    """
    # If the layer is not a container, forward pass.
    if (len(list(model.children())) == 0):
        return model(x), layer_index - 1
    else:  # Recurse otherwise.
        for sublayer in model.children():
            x, layer_index = truncated_model(x, sublayer, layer_index)
            if layer_index < 0:  # Stop at the specified layer.
                return x, layer_index

def weighted_cumulate(new_bar, bar_sum, unit, response):
    """
    Adds the new_bar, weighted by the unit's response to that bar, cumulative
    bar map.
    
    Parameters
    ----------
    new_bar : numpy.array
        The new bar.
    bar_sum : numpy.array
        The cumulated weighted sum of all previous bars. This is modified
        in-place.
    unit : int
        The unit's number.
    response : float
        The unit's response (spatial center only) to the new bar.
    """
    bar_sum[unit, :, :] += new_bar * response

def threshold_cumulate(new_bar, bar_sum, unit, response, threshold):
    """
    Adds to a cumulative map only bars that gave a threshold response.
    
    Parameters
    ----------
    See weighted_cumulate() for repeated parameters.
    threshold : float
        The unit's response (spatial center only) to the new bar.
    """
    if response > threshold:
        bar_sum[unit, :, :] += new_bar

def center_only_cumulate(center_index, bar_sum, unit, response, threshold):
    """
    Add to a cumulative map only the center points of bars that gave a
    threshold response.
    
    Parameters
    ----------
    See weighted_cumulate() for repeated parameters.
    center_index : (int, int)
        The center of the bar. 
    bar_sum : numpy.array
        The cumulated weighted sum of all previous bar centers. This is
        modified in-place.
    """
    if response > threshold:
        bar_sum[unit, center_index[0], center_index[1]] += response
        
def print_progress(num_stimuli):
    sys.stdout.write('\r')
    sys.stdout.write(f"num_stimuli = {num_stimuli}")
    sys.stdout.flush()

delete_all_npy_files(result_dir)
# rf_blen_ratio, aspect_ratio, theta, fgval, bgval
for conv_i, layer_index in enumerate(layer_indices):
    layer_name = f"conv{conv_i + 1}"
    num_units = nums_units[conv_i]
    rf_size = rf_sizes[conv_i][0]
    print(f"\nAnalyzing {layer_name}...")

    # Get spatial center and the corresponding box in pixel space.
    spatial_index = np.array(conv_output_shapes[conv_i][-2:])
    spatial_index = calculate_center(spatial_index)
    box = converter.convert(spatial_index, layer_index, 0, is_forward=False)
    xc, yc = box_to_center(box)

    # Initializations
    num_stimuli = 0
    weighted_bar_sum = np.zeros((num_units, yn, xn))
    threshold_bar_sum = np.zeros((num_units, yn, xn))
    center_only_bar_sum = np.zeros((num_units, yn, xn))
    unit_blen_bwid_theta_val_responses = np.zeros((num_units,
                                                   len(rf_blen_ratios),
                                                   len(aspect_ratios),
                                                   len(thetas),
                                                   2))

    for blen_i, rf_blen_ratio in enumerate(rf_blen_ratios):
        for bwid_i, aspect_ratio in enumerate(aspect_ratios):
            for theta_i, theta in enumerate(thetas):
                for val_i, (fgval, bgval) in enumerate([(1, -1), (-1, 1)]):
                    # Some bar parameters
                    blen = round(rf_blen_ratio * rf_size)
                    bwid = round(aspect_ratio * blen)
                    grid_spacing = blen/2
                    
                    # Get grid coordinates.
                    grid_coords = bar_locations.get_grid_coords(layer_index, spatial_index, grid_spacing)
                    grid_coords_np = np.array(grid_coords)

                    # Create bars.
                    for grid_coord_i, (xc, yc) in enumerate(grid_coords_np):
                        if this_is_a_test_run and grid_coord_i > 10:
                            break
                        
                        bar = draw_bar(xn, yn, xc, yc, theta, blen, bwid, laa, fgval, bgval)
                        bar_tensor = preprocess_img_to_tensor(bar)
                        y, _ = truncated_model(bar_tensor, model, layer_index)
                        center_responses = y[0, :, spatial_index[0], spatial_index[1]].cpu().detach().numpy()
                        center_responses[center_responses < 0] = 0  # ReLU
                        unit_blen_bwid_theta_val_responses[:, blen_i, bwid_i, theta_i, val_i] += center_responses[:]
                        num_stimuli += 1
                        print_progress(num_stimuli)

                        for unit in range(num_units):
                            weighted_cumulate(bar, weighted_bar_sum, unit, center_responses[unit])
                            threshold_cumulate(bar, threshold_bar_sum, unit, center_responses[unit], threshold)
                            center_only_cumulate((yc, xc), center_only_bar_sum, unit, center_responses[unit], threshold)

    weighted_map_path = os.path.join(result_dir, f"{layer_name}.weighted.cumulative_map.npy")
    threshold_map_path = os.path.join(result_dir, f"{layer_name}.threshold.cumulative_map.npy")
    center_only_map_path = os.path.join(result_dir, f"{layer_name}.center_only.cumulative_map.npy")
    np.save(weighted_map_path, weighted_bar_sum)
    np.save(threshold_map_path, threshold_bar_sum)
    np.save(center_only_map_path, center_only_bar_sum)

    cumulative_tuning_path = os.path.join(result_dir, f"{layer_name}.cumulative_tuning.npy")
    np.save(cumulative_tuning_path, unit_blen_bwid_theta_val_responses)

    for cumulate_mode, bar_sum in zip(['weighted', 'threshold', 'center_only'],
                                      [weighted_bar_sum, threshold_bar_sum, center_only_bar_sum]):
        cumulative_pdf_path = os.path.join(result_dir, f"{layer_name}.{cumulate_mode}.cumulative.pdf")
        with PdfPages(cumulative_pdf_path) as pdf:
            for unit in range(num_units):
                plt.figure(figsize=(25, 5))
                plt.suptitle(f"RF mapping with bars (no.{unit}, {num_stimuli} stimuli)", fontsize=20)
                
                plt.subplot(1, 5, 1)
                plt.imshow(bar_sum[unit, :, :], cmap='gray')
                plt.title("Cumulated bar maps")
                boundary = 10
                plt.xlim([box[1] - boundary, box[3] + boundary])
                plt.ylim([box[0] - boundary, box[2] + boundary])
                rect = make_box(box, linewidth=2)
                ax = plt.gca()
                ax.add_patch(rect)
                ax.invert_yaxis()
                
                plt.subplot(1, 5, 2)
                blen_tuning = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(1,2,3))
                blen_std = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(1,2,3))/math.sqrt(num_units)
                plt.errorbar(rf_blen_ratios, blen_tuning, yerr=blen_std)
                plt.title("Bar length tuning")
                plt.xlabel("blen/RF ratio")
                plt.ylabel("avg response")
                plt.grid()
                
                plt.subplot(1, 5, 3)
                bwid_tuning = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,2,3))
                bwid_std = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,2,3))/math.sqrt(num_units)
                plt.errorbar(aspect_ratios, bwid_tuning, yerr=bwid_std)
                plt.title("Bar width tuning")
                plt.xlabel("aspect ratio")
                plt.ylabel("avg response")
                plt.grid()

                plt.subplot(1, 5, 4)
                theta_tuning = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,1,3))
                theta_std = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,1,3))/math.sqrt(num_units)
                plt.errorbar(thetas, theta_tuning, yerr=theta_std)
                plt.title("Theta tuning")
                plt.xlabel("theta")
                plt.ylabel("avg response")
                plt.grid()
                
                plt.subplot(1, 5, 5)
                val_tuning = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,1,2))
                val_std = np.mean(unit_blen_bwid_theta_val_responses[unit,...], axis=(0,1,2))/math.sqrt(num_units)
                plt.bar(['white on black', 'black on white'], val_tuning, yerr=val_std, width=0.4)
                plt.title("Contrast tuning")
                plt.ylabel("avg response")
                plt.grid()

                pdf.savefig()
                plt.close()
