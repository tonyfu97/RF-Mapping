"""
Show the network on block of color at a time. This is very similar to occluder
discrepancy mappping, but the background is made of zeros.

Oct 11, 2022
Tony Fu
"""
import os
import sys
import math
import warnings

import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../..')
from src.rf_mapping.image import make_box
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.spatial import (xn_to_center_rf,
                                    calculate_center,
                                    get_rf_sizes,)
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c
from src.rf_mapping.stimulus import *
from src.rf_mapping.occluder_discrepancy import get_occluder_params

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take hours to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")


#######################################.#######################################
#                                                                             #
#                             DRAW_COLOR_BLOCK                                #
#                                                                             #
###############################################################################
def make_color_block(shape, top_left, bottom_right, block_color,
                     background_color=(0.0, 0.0, 0.0)):
    """
    Returns a square on a background.
    Note: the y-axis points downward.

    Parameters
    ----------
    shape: (int, int)
        The shape of the entire torch tensor (not including the color channel).
    top_left: (int, int)
        Spatial index of the top left corner (inclusive) of the block.
    bottom_right: (int, int)
        Spatial index of the bottom right corner (inclusive) of the block.
    block_color: (float, float, float)
        The RGB color of the block between [-1, 1].
    background_color: (float, float, float)
        The RGB color of the background between [-1, 1].

    Returns
    -------
    block_img_tensor : torch.tensor
        img but with color block at the specified location.
    """
    output_tensor = torch.empty((3, *shape))
    for color_i in range(3):
        output_tensor[color_i] = background_color[color_i]
        output_tensor[color_i, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = block_color[color_i]
    return output_tensor.detach()


# Plot an example block.
if __name__ == "__main__":
    block_img = make_color_block((200, 150), (25, 30), (150, 80), (1,-1,-1), (0, 0, 0))
    plt.imshow(np.transpose(block_img.numpy(), (1,2,0)))
    plt.show()


#######################################.#######################################
#                                                                             #
#                               GET_BLOCK_PARAMS                              #
#                                                                             #
###############################################################################
def get_block_params(rf_size, xn):
    """
    Creates a list of parameters for the make_color_block() function.
    
    Parameters
    ----------
    rf_size: int
        Side length of RF (assumed to be square). Sometimes, the box may be
        cropped, so we will use rf_size instead of the size of the 'box' to
        calculate the occluder size and stride.
    xn: int
        The size of the block image. Must be >= rf_size.
    
    Returns
    -------
    block_params: [{str: (int, int), str: (int, int), str: (float, float, float)}, ...]
        A list of parameters. Each one is a dictionary with the parameter names
        and values.
    """
    block_size = max(rf_size // 20, 1)
    block_stride = max(block_size // 3, 1)
    padding = (xn - rf_size) // 2
    
    v1 = 1.0
    v0 = -1.0
    color_list = [(v1, v1, v1),
                  (v0, v0, v0),
                  (v1, v0, v0),
                  (v0, v1, v0),
                  (v0, v0, v1),
                  (v1, v1, v0),
                  (v0, v1, v1),
                  (v1, v0, v1),]

    block_params = []

    for i in np.arange(padding, xn-padding-1, block_stride):
        for j in np.arange(padding, xn-padding-1, block_stride):
            for color in color_list:
                block_params.append({'top_left' : (i, j), 
                                     'bottom_right' : (i+padding-1, j+padding-1),
                                     'block_color': color})
    return block_params


#######################################.#######################################
#                                                                             #
#                           BLOCK_MAP_CENTER_RESPONSES                        #
#                                                                             #
###############################################################################
def block_map_center_resposnes(xn, block_params, truncated_model, num_units,
                               batch_size=100, _debug=False):
    """
    Presents blocks and returns the center responses in array of dimension:
    [num_stim, num_units].

    Parameters
    ----------
    xn - the size of each block image.\n
    block_params    - block stimulus parameter list.\n
    truncated_model - neural network up to the layer of interest.\n
    num_units  - number of units/channels.\n
    batch_size - how many blocks to present at once.\n
    _debug     - if true, reduce the number of blocks and plot them.\n
    """
    block_i = 0
    num_stim = len(block_params)
    center_responses = np.zeros((num_stim, num_units))

    while (block_i < num_stim):
        if _debug and block_i > 200:
            break
        print_progress(f"Presenting {block_i}/{num_stim} stimuli...")
        real_batch_size = min(batch_size, num_stim-block_i)
        block_batch = torch.zeros((real_batch_size, 3, xn, xn))

        # Create a batch of blocks.
        for i in range(real_batch_size):
            params = block_params[block_i + i]
            block_batch[i] = make_color_block((xn, xn),
                                            params['top_left'],
                                            params['bottom_right'],
                                            params['block_color'])

        # Present the patch of blocks to the truncated model.
        with torch.no_grad():  # turn off gradient calculations for speed.
            y = truncated_model(block_batch.to(c.DEVICE))
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[block_i:block_i+real_batch_size, :] =\
                                    y[:, :, yc, xc].detach().cpu().numpy()
        block_i += real_batch_size

    return center_responses


#######################################.#######################################
#                                                                             #
#                               MAKE_BLOCK_MAPS                               #
#                                                                             #
###############################################################################
def make_block_maps(xn, block_params, center_responses, unit_i, _debug=False, 
                    response_thr=0.5):
    """
    Parameters
    ----------
    xn - the size of each block image.\n
    block_params     - block stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n
    unit_i           - unit's index.\n
    response_thr     - block w/ a reponse below response_thr * rmax will be
                       excluded.\n
    _debug           - if true, print ranking info.\n

    Returns
    -------
    The weighted_max_map, weighted_min_map, non_overlap_max_map, and
    non_overlap_min_map of one unit.
    """
    print(f"{unit_i} done.")
    
    weighted_max_map = np.zeros((3, xn, xn))
    weighted_min_map = np.zeros((3, xn, xn))

    isort = np.argsort(center_responses[:, unit_i])  # Ascending
    r_max = center_responses[:, unit_i].max()
    r_min = center_responses[:, unit_i].min()
    r_range = max(r_max - r_min, 1)

    # Initialize block counts
    num_weighted_max_blocks = 0
    num_weighted_min_blocks = 0

    if _debug:
        print(f"unit {unit_i}: r_max: {r_max:7.2f}, max block idx: {isort[::-1][:5]}")

    for max_block_i in isort[::-1]:
        response = center_responses[max_block_i, unit_i]
        params = block_params[max_block_i]
        # Note that the background color are set to 0, while the foreground
        # values are always positive.
        if params['block_color'] == (-1, -1, -1):
            block_color = (1, 1, 1)
        else:
            block_color = [max(0, color) for color in params['block_color']]
        new_block = make_color_block((xn, xn),
                                     params['top_left'],
                                     params['bottom_right'],
                                     block_color).detach().cpu().numpy()
        if (response > max(r_max * response_thr, 0)):
            add_weighted_map(new_block, weighted_max_map, response)
            # counts the number of blocks in each map
            num_weighted_max_blocks += 1
        else:
            break

    for min_block_i in isort:
        response = center_responses[min_block_i, unit_i]
        params = block_params[min_block_i]
        if params['block_color'] == (-1, -1, -1):
            block_color = (1, 1, 1)
        else:
            block_color = [max(0, color) for color in params['block_color']]
        new_block = make_color_block((xn, xn),
                                     params['top_left'],
                                     params['bottom_right'],
                                     block_color).detach().cpu().numpy()
        if (response < min(r_min * response_thr, 0)): 
            add_weighted_map(new_block, weighted_min_map, -response)
            # counts the number of blocks in each map
            num_weighted_min_blocks += 1
        else:
            break

    return weighted_max_map, weighted_min_map,\
           num_weighted_max_blocks, num_weighted_min_blocks


#######################################.#######################################
#                                                                             #
#                                BLOCK_MAP_RUN                                #
#                                                                             #
###############################################################################
def block_map_run(model, model_name, result_dir, _debug=False, batch_size=100,
                  response_thres=0.5):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4c7o,
    which is like paradigm 4a, but with 6 additional colors.
    
    Parameters
    ----------
    model      - neural network.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    delete_all_npy_files(result_dir)
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image_size upper bounds are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during block mapping.
    
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_block_weighted_counts.txt")
    if os.path.exists(weighted_counts_path):
        os.remove(weighted_counts_path)
    
    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        block_params = get_block_params(max_rf, xn)
        
        # Record this run into the log.
        log_path = os.path.join(result_dir, f"script_log.txt")
        record_script_log(log_path, layer_name, batch_size, response_thres, len(block_params))

        # Array initializations
        weighted_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all blocks to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = block_map_center_resposnes(xn, block_params,
                                        truncated_model, num_units,
                                        batch_size=batch_size,
                                        _debug=_debug)

        # make_block_maps(xn, block_params, center_responses, 0, _debug=False, 
        #                 response_thr=response_thres)
        unit_batch_size = os.cpu_count() // 3
        unit_i = 0
        while (unit_i < num_units):
            if _debug and unit_i >= 20:
                break
            real_batch_size = min(unit_batch_size, num_units - unit_i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(make_block_maps,
                            [xn for _ in range(real_batch_size)],
                            [block_params for _ in range(real_batch_size)],
                            [center_responses for _ in range(real_batch_size)],
                            [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                            [_debug for _ in range(real_batch_size)],
                            [response_thres for _ in range(real_batch_size)],
                            )
            # Crop and save maps to layer-level array
            for result_i, result in enumerate(results):
                weighted_max_maps[unit_i + result_i] = result[0][:,padding:padding+max_rf, padding:padding+max_rf]
                weighted_min_maps[unit_i + result_i] = result[1][:,padding:padding+max_rf, padding:padding+max_rf]
                # Record the number of bars used in each map (append to txt files).
                record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                  result[2], result[3])
            unit_i += real_batch_size

        # Save the maps of all units.
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_blockmaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_blockmaps.npy")
        np.save(weighte_max_maps_path, weighted_max_maps)
        np.save(weighted_min_maps_path, weighted_min_maps)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        top_n = min(5000, len(block_params)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top{top_n}_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot{top_n}_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)
        
        # Make pdf for the layer.
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_blockmaps.pdf")
        make_map_pdf(np.transpose(weighted_max_maps, (0,2,3,1)),
                     np.transpose(weighted_min_maps, (0,2,3,1)),
                     weighted_pdf_path)
