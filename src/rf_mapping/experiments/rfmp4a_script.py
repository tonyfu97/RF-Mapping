"""
Receptive field mapping paradigm 4a.

Note: all code assumes that the y-axis points downward.

Tony Fu, July 4th, 2022
"""
import os
import sys

import numpy as np
from pyparsing import nums
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from spatial import (clip,
                     get_conv_output_shapes,
                     calculate_center,
                     get_rf_sizes,
                     RfGrid,
                     SpatialIndexConverter,)
from image import make_box, preprocess_img_to_tensor
from hook import ConvUnitCounter
from stimulus import draw_bar
import constants as c

# Please specify some details here:
model = models.alexnet()
model_name = 'alexnet'
xn = yn = 227
spatial_index = (11, 11)
rf_blen_ratios = [3/4, 3/8, 3/16, 3/32]
rf_blen_ratio_strs = ['3/4', '3/8', '3/16', '3/32']
aspect_ratios = [1/2, 1/5, 1/10]
thetas = np.arange(0, 180, 22.5)
laa = 0.5
fgval = 1.0
bgval = 0.5
cumulate_mode = 'threshold'  #['weighted', 'threshold', 'center_only']
threshold = 1

# Please double-check the directories:
pdf_dir = c.REPO_DIR + f'/results/rfmp4a/{model_name}/'
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


# rf_blen_ratio, aspect_ratio, theta, fgval, bgval
for conv_i, layer_index in enumerate(layer_indices):
    layer_name = f"conv{conv_i + 1}"
    num_units = nums_units[conv_i]
    # Get spatial center of box.
    spatial_index = np.array(conv_output_shapes[conv_i][-2:])
    spatial_index = calculate_center(spatial_index)
    box = converter.convert(spatial_index, layer_index, 0, is_forward=False)
    
    # Initialize bar sum.
    bar_sum = np.zeros((num_units, yn, xn))
    num_stimuli = 0
                    
    for i, rf_blen_ratio in enumerate(rf_blen_ratios):
        for aspect_ratio in aspect_ratios:
            for theta in thetas:
                for fgval, bgval in [(1, -1), (-1, 1)]:
                    # Some bar parameters
                    rf_size = rf_sizes[conv_i][0]
                    blen = round(rf_blen_ratio * rf_size)
                    bwid = round(aspect_ratio * blen)
                    grid_spacing = blen/2
                    
                    # Get grid coordinates.
                    grid_coords = bar_locations.get_grid_coords(layer_index, spatial_index, grid_spacing)
                    grid_coords_np = np.array(grid_coords)
                    
                    # Create bars.
                    print(f"Conv{conv_i+1}: using {len(grid_coords)} number of grid points...")
                    for xc, yc in tqdm(grid_coords_np):
                        bar = draw_bar(xn, yn, xc, yc, theta, blen, bwid, laa, fgval, bgval)
                        bar_tensor = preprocess_img_to_tensor(bar)
                        y, _ = truncated_model(bar_tensor, model, layer_index)
                        center_responses = y[0, :, spatial_index[0], spatial_index[1]].cpu().detach().numpy()
                        center_responses[center_responses < 0] = 0
                        num_stimuli += 1

                        for unit in range(num_units):
                            if cumulate_mode == 'weighted':
                                weighted_cumulate(bar, bar_sum, unit, center_responses[unit])
                            elif cumulate_mode == 'threshold':
                                threshold_cumulate(bar, bar_sum, unit, center_responses[unit], threshold)
                            elif cumulate_mode == 'center_only':
                                center_only_cumulate(spatial_index, bar_sum, unit, center_responses[unit], threshold)
                            else:
                                raise ValueError(f"cumulate mode '{cumulate_mode}' is not supported.")

    cumulative_pdf_path = os.path.join(pdf_dir, f"{layer_name}.{cumulate_mode}.cumulative.pdf")
    with PdfPages(cumulative_pdf_path) as pdf:
        for unit in range(num_units):
            plt.imshow(bar_sum[unit, :, :], cmap='gray')
            plt.title(f"no.{unit}, Bar Length = {rf_blen_ratio_strs[i]} M, aspect_ratio = {aspect_ratio}, theta = {theta}")
            
            boundary = 10
            plt.xlim([box[1] - boundary, box[3] + boundary])
            plt.ylim([box[0] - boundary, box[2] + boundary])
            
            rect = make_box(box, linewidth=2)
            ax = plt.gca()
            ax.add_patch(rect)
            ax.invert_yaxis()
            
            pdf.savefig()
            plt.show()
            plt.close()
    print(f"number of stimuli = {num_stimuli} per unit")