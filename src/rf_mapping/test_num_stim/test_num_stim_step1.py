"""
Find the good number of bars used for our mapping methods.

Step 1. Map the RF using the specificied numbers of stimuli.

THIS SCRIPT TAKES A LOT OF TIME TO RUN!

Tony Fu, August 18th, 2022
"""
from doctest import REPORT_NDIFF
import os
import sys
from time import time

import concurrent.futures
import numpy as np
import torch.nn as nn
import torchvision.models as models

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.stimulus import *
import src.rf_mapping.bar as bar
import src.rf_mapping.grating as grating
import src.rf_mapping.pasu_shape as pasu_shape
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.spatial import (xn_to_center_rf,
                                    get_rf_sizes,)

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
this_is_a_test_run = False
batch_size = 10
conv_i_to_run = 1  # conv_i = 1 means Conv2
rfmp_name = 'rfmp4a'
num_stim_list = [50, 100, 250, 500, 750, 1000, 1500, 2000, 5000]
response_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

result_dir = os.path.join(c.REPO_DIR, 'results', 'test_num_stim', rfmp_name)

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else:
#         raise KeyboardInterrupt("Interrupted by user")

layer_name = f"conv{conv_i_to_run+1}"


################################     RFMP4a    ################################
def make_barmaps(splist, center_responses, unit_i, _debug=False, has_color=False, 
                 num_bars=500, response_thr=0.8, stim_thr=0.2):
    """
    Parameters
    ----------
    splist           - bar stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n
    unit_i           - unit's index.\n
    response_thr     - bar w/ a reponse below response_thr * rmax will be
                       excluded.\n
    stim_thr         - bar pixels w/ a value below stim_thr will be excluded.\n
    _debug           - if true, print ranking info.\n

    Returns
    -------
    The weighted_max_map, weighted_min_map, non_overlap_max_map, and
    non_overlap_min_map of one unit.
    
    TODO: This function is a bottleneck. Must uses multiprocessing to
          parallelize the computations on multiple cpu cores.
    """
    print(f"{unit_i} done.")

    xn = splist[0]['xn']
    yn = splist[0]['yn']
    
    if has_color:
        weighted_max_map = np.zeros((3, yn, xn))
        weighted_min_map = np.zeros((3, yn, xn))
        non_overlap_max_map = np.zeros((3, yn, xn))
        non_overlap_min_map = np.zeros((3, yn, xn))
    else:
        weighted_max_map = np.zeros((yn, xn))
        weighted_min_map = np.zeros((yn, xn))
        non_overlap_max_map = np.zeros((yn, xn))
        non_overlap_min_map = np.zeros((yn, xn))

    isort = np.argsort(center_responses[:, unit_i])  # Ascending
    r_max = center_responses[:, unit_i].max()
    r_min = center_responses[:, unit_i].min()
    r_range = max(r_max - r_min, 1)

    # Initialize bar counts
    num_weighted_max_bars = 0
    num_weighted_min_bars = 0
    num_non_overlap_max_bars = 0
    num_non_overlap_min_bars = 0

    if _debug:
        print(f"unit {unit_i}: r_max: {r_max:7.2f}, max bar idx: {isort[::-1][:5]}")

    for max_bar_i in isort[::-1]:
        response = center_responses[max_bar_i, unit_i]
        params = splist[max_bar_i]
        # Note that the background color are set to 0, while the foreground
        # values are always positive.
        if has_color:
            new_bar = bar.stimfr_bar_color(params['xn'], params['yn'],
                                           params['x0'], params['y0'],
                                           params['theta'],
                                           params['len'], params['wid'], 
                                           params['aa'],
                                           params['r1'], params['g1'], params['b1'],
                                           params['r0'], params['g0'], params['b0'])
        else:
            new_bar = bar.stimfr_bar(params['xn'], params['yn'],
                                params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'],
                                0.5, 1, 0)
        # if (response - r_min) > r_range * response_thr:
        # if num_weighted_max_bars < num_bars:
        if (response > max(r_max * response_thr, 0)) and num_weighted_max_bars < 5000:
            # has_included = add_non_overlap_map(new_bar, non_overlap_max_map, stim_thr)
            # add_weighted_map(new_bar, weighted_max_map, (response - r_min)/r_range)
            add_weighted_map(new_bar, weighted_max_map, response)
            # counts the number of bars in each map
            num_weighted_max_bars += 1
            # if has_included:
            #     num_non_overlap_max_bars += 1
        else:
            break

    for min_bar_i in isort:
        response = center_responses[min_bar_i, unit_i]
        params = splist[min_bar_i]
        if has_color:
            new_bar = bar.stimfr_bar_color(params['xn'], params['yn'],
                                           params['x0'], params['y0'],
                                           params['theta'],
                                           params['len'], params['wid'], 
                                           params['aa'],
                                           params['r1'], params['g1'], params['b1'],
                                           params['r0'], params['g0'], params['b0'])
        else:
            new_bar = bar.stimfr_bar(params['xn'], params['yn'],
                                params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'],
                                0.5, 1, 0) 
        # if (response - r_min) < r_range * (1 - response_thr):
        # if num_weighted_min_bars < num_bars:
        if (response < min(r_min * response_thr, 0)) and num_weighted_min_bars < 5000:
            # has_included = add_non_overlap_map(new_bar, non_overlap_min_map, stim_thr)
            # add_weighted_map(new_bar, weighted_min_map, (r_max - response)/r_range)
            add_weighted_map(new_bar, weighted_min_map, -response)
            # counts the number of bars in each map
            num_weighted_min_bars += 1
            # if has_included:
            #     num_non_overlap_min_bars += 1
        else:
            break

    return weighted_max_map, weighted_min_map,\
           non_overlap_max_map, non_overlap_min_map,\
           num_weighted_max_bars, num_weighted_min_bars,\
           num_non_overlap_max_bars, num_non_overlap_min_bars


def rfmp4a_run_01b_test_num_stim(model, model_name, result_dir, _debug=False, batch_size=100,
                                 num_bars_list=[], conv_i_to_run=None, response_thresholds=[]):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4a.
    
    Parameters
    ----------
    model      - neural network.\n
    model_name - name of neural network. Used for txt file naming.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image sizes above are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb100.txt")
    
    # Delete previous files
    delete_all_npy_files(result_dir)
    if os.path.exists(tb1_path):
        os.remove(tb1_path)
    if os.path.exists(tb20_path):
        os.remove(tb20_path)
    if os.path.exists(tb100_path):
        os.remove(tb100_path)
    
    for conv_i in range(len(layer_indices)):
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue

        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = bar.stimset_dict_rfmp_4a(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = bar.barmap_run_01b(splist, truncated_model,
                                          num_units, batch_size=batch_size,
                                          _debug=_debug)

        # Append to txt files that summarize the top and bottom bars.
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)

        # Save the splist in a text file.
        splist_path = os.path.join(result_dir, f"{layer_name}_splist.txt")
        if os.path.exists(splist_path):
            os.remove(splist_path)
        record_splist(splist_path, splist)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top{top_n}_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot{top_n}_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)
        
        # for num_bars in num_bars_list:
        for response_thr in response_thresholds:
            num_bars = 0
            start = time()
            
            # num_stim_result_dir = os.path.join(result_dir, str(num_bars))
            num_stim_result_dir = os.path.join(result_dir, str(response_thr))
            if not os.path.exists(num_stim_result_dir):
                os.makedirs(num_stim_result_dir)
            print(f"Result directory: {num_stim_result_dir}...")
            weighted_counts_path = os.path.join(num_stim_result_dir,
                                                f"{model_name}_rfmp4a_weighted_counts.txt")
            non_overlap_counts_path = os.path.join(num_stim_result_dir,
                                                   f"{model_name}_rfmp4a_non_overlap_counts.txt")
            if os.path.exists(weighted_counts_path):
                os.remove(weighted_counts_path)
            if os.path.exists(non_overlap_counts_path):
                os.remove(non_overlap_counts_path)

            # This block of code contained in the following while-loop used to be
            # a bottleneck of the program because it is all computed by a single
            # CPU core. Improvement by multiprocessing was implemented on August
            # 15, 2022 to solve the problem.
            batch_size = os.cpu_count() // 2
            unit_i = 0
            while (unit_i < num_units):
                if _debug and unit_i >= 20:
                    break
                real_batch_size = min(batch_size, num_units - unit_i)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(make_barmaps,
                                        [splist for _ in range(real_batch_size)],
                                        [center_responses for _ in range(real_batch_size)],
                                        [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                        [_debug for _ in range(real_batch_size)],
                                        [False for _ in range(real_batch_size)],
                                        [num_bars for _ in range(real_batch_size)],
                                        [response_thr for _ in range(real_batch_size)],
                                        )
                # Crop and save maps to layer-level array
                for result_i, result in enumerate(results):
                    weighted_max_maps[unit_i + result_i] = result[0][padding:padding+max_rf, padding:padding+max_rf]
                    weighted_min_maps[unit_i + result_i] = result[1][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_max_maps[unit_i + result_i] = result[2][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_min_maps[unit_i + result_i] = result[3][padding:padding+max_rf, padding:padding+max_rf]
                    # Record the number of bars used in each map (append to txt files).
                    record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                    result[4], result[5])
                    # record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                    #                 result[6], result[7])
                unit_i += real_batch_size

            # Save the maps of all units.
            weighte_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_max_barmaps.npy")
            weighted_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_min_barmaps.npy")
            # non_overlap_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_max_barmaps.npy")
            # non_overlap_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_min_barmaps.npy")
            np.save(weighte_max_maps_path, weighted_max_maps)
            np.save(weighted_min_maps_path, weighted_min_maps)
            # np.save(non_overlap_max_maps_path, non_overlap_max_maps)
            # np.save(non_overlap_min_maps_path, non_overlap_min_maps)

            # Make pdf for the layer.
            weighted_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_barmaps.pdf")
            make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
            # non_overlap_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_barmaps.pdf")
            # make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)
            
            end = time()

            total_time = end - start        
            
            speed_txt_path = os.path.join(result_dir, f"speed.txt")
            with open(speed_txt_path, 'a') as f:
                # f.write(f"{num_bars} {total_time}\n")
                f.write(f"{response_thr} {total_time}\n")



################################    RFMP4c7o    ###############################


def rfmp4c7o_run_01_test_num_stim(model, model_name, result_dir, _debug=False, batch_size=100,
                                  num_bars_list=[], conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4c7o,
    which is like paradigm 4a, but with 6 additional colors.
    
    Parameters
    ----------
    model      - neural network.\n
    model_name - name of neural network. Used for txt file naming.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image_size upper bounds are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb100.txt")

    # Delete previous files
    delete_all_npy_files(result_dir)
    if os.path.exists(tb1_path):
        os.remove(tb1_path)
    if os.path.exists(tb20_path):
        os.remove(tb20_path)
    if os.path.exists(tb100_path):
        os.remove(tb100_path)
    
    for conv_i in range(len(layer_indices)):
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue
        
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = bar.stimset_dict_rfmp_4c7o(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        # non_overlap_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        # non_overlap_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = bar.barmap_run_01b(splist, truncated_model,
                                          num_units, batch_size=batch_size,
                                          _debug=_debug, has_color=True)

        # Append to txt files that summarize the top and bottom bars.
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)
        
        # Save the splist in a text file.
        splist_path = os.path.join(result_dir, f"{layer_name}_splist.txt")
        if os.path.exists(splist_path):
            os.remove(splist_path)
        record_splist(splist_path, splist)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        # Note: the splist and center responses are recorded as text files
        #       because we are interested in reducing the number of bar
        #       stimuli. We intend to drop the small bars if they are not
        #       commonly found as the top and bottom N bars.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top5000_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot5000_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)
        
        for num_bars in num_bars_list:
            
            start = time()
            
            num_stim_result_dir = os.path.join(result_dir, str(num_bars))
            if not os.path.exists(num_stim_result_dir):
                os.makedirs(num_stim_result_dir)
            print(f"Result directory: {num_stim_result_dir}...")
            weighted_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_weighted_counts.txt")
            # non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_non_overlap_counts.txt")
            if os.path.exists(weighted_counts_path):
                os.remove(weighted_counts_path)
            # if os.path.exists(non_overlap_counts_path):
            #     os.remove(non_overlap_counts_path)

            # This block of code contained in the following while-loop used to be
            # a bottleneck of the program because it is all computed by a single
            # CPU core. Improvement by multiprocessing was implemented on August
            # 15, 2022 to solve the problem.
            batch_size = os.cpu_count() // 2
            unit_i = 0
            while (unit_i < num_units):
                if _debug and unit_i >= 20:
                    break
                real_batch_size = min(batch_size, num_units - unit_i)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(make_barmaps,
                                        [splist for _ in range(real_batch_size)],
                                        [center_responses for _ in range(real_batch_size)],
                                        [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                        [_debug for _ in range(real_batch_size)],
                                        [True for _ in range(real_batch_size)],
                                        [num_bars for _ in range(real_batch_size)],
                                        )
                # Crop and save maps to layer-level array
                for result_i, result in enumerate(results):
                    weighted_max_maps[unit_i + result_i] = result[0][:,padding:padding+max_rf, padding:padding+max_rf]
                    weighted_min_maps[unit_i + result_i] = result[1][:,padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_max_maps[unit_i + result_i] = result[2][:,padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_min_maps[unit_i + result_i] = result[3][:,padding:padding+max_rf, padding:padding+max_rf]
                    # Record the number of bars used in each map (append to txt files).
                    record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                    result[4], result[5])
                    # record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                    #                 result[6], result[7])
                unit_i += real_batch_size

            # Save the maps of all units.
            weighte_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_max_barmaps.npy")
            weighted_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_min_barmaps.npy")
            # non_overlap_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_max_barmaps.npy")
            # non_overlap_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_min_barmaps.npy")
            np.save(weighte_max_maps_path, weighted_max_maps)
            np.save(weighted_min_maps_path, weighted_min_maps)
            # np.save(non_overlap_max_maps_path, non_overlap_max_maps)
            # np.save(non_overlap_min_maps_path, non_overlap_min_maps)

            
            # Make pdf for the layer.
            weighted_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_barmaps.pdf")
            make_map_pdf(np.transpose(weighted_max_maps, (0,2,3,1)),
                        np.transpose(weighted_min_maps, (0,2,3,1)),
                        weighted_pdf_path)
            # non_overlap_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_barmaps.pdf")
            # make_map_pdf(np.transpose(non_overlap_max_maps, (0,2,3,1)),
            #             np.transpose(non_overlap_min_maps, (0,2,3,1)),
            #             non_overlap_pdf_path)

            end = time()

            total_time = end - start        
            
            speed_txt_path = os.path.join(result_dir, f"speed.txt")
            with open(speed_txt_path, 'a') as f:
                f.write(f"{num_bars} {total_time}\n")


###############################    RFMP_SIN1    ###############################


def sin1_run_01b_test_num_stim(model, model_name, result_dir, _debug=False, batch_size=10,
                               num_stim_list=[], conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm sin1.
    
    Parameters
    ----------
    model      - neural network.\n
    model_name - name of neural network. Used for txt file naming.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image sizes above are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_sin1_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_sin1_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_sin1_tb100.txt")
    
    # Delete previous files
    delete_all_npy_files(result_dir)
    if os.path.exists(tb1_path):
        os.remove(tb1_path)
    if os.path.exists(tb20_path):
        os.remove(tb20_path)
    if os.path.exists(tb100_path):
        os.remove(tb100_path)
    
    for conv_i in range(len(layer_indices)):
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue
        
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = grating.stimset_dict_rfmp_sin1(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf) // 2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = grating.sinmap_run_01b(splist, truncated_model, num_units,
                                          batch_size=batch_size, _debug=False)

        # Append to txt files that summarize the top and bottom bars.
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)
        
        # Save the splist in a text file.
        splist_path = os.path.join(result_dir, f"{layer_name}_splist.txt")
        if os.path.exists(splist_path):
            os.remove(splist_path)
        record_splist(splist_path, splist)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top{top_n}_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot{top_n}_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)
        
        for num_stim in num_stim_list:
            
            start = time()

            num_stim_result_dir = os.path.join(result_dir, str(num_stim))
            if not os.path.exists(num_stim_result_dir):
                os.makedirs(num_stim_result_dir)
            print(f"Result directory: {num_stim_result_dir}...")
            weighted_counts_path = os.path.join(result_dir, f"{model_name}_sin1_weighted_counts.txt")
            # non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_sin1_non_overlap_counts.txt")
            if os.path.exists(weighted_counts_path):
                os.remove(weighted_counts_path)
            # if os.path.exists(non_overlap_counts_path):
            #     os.remove(non_overlap_counts_path)

            # This block of code contained in the following while-loop used to be
            # a bottleneck of the program because it is all computed by a single
            # CPU core. Improvement by multiprocessing was implemented on August
            # 15, 2022 to solve the problem.
            batch_size = os.cpu_count() // 2
            unit_i = 0
            while (unit_i < num_units):
                if _debug and unit_i >= 20:
                    break
                real_batch_size = min(batch_size, num_units - unit_i)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(grating.make_stimmaps,
                                        [splist for _ in range(real_batch_size)],
                                        [center_responses for _ in range(real_batch_size)],
                                        [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                        [_debug for _ in range(real_batch_size)],
                                        [num_stim for _ in range(real_batch_size)],
                                        )
                # Crop and save maps to layer-level array
                for result_i, result in enumerate(results):
                    weighted_max_maps[unit_i + result_i] = result[0][padding:padding+max_rf, padding:padding+max_rf]
                    weighted_min_maps[unit_i + result_i] = result[1][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_max_maps[unit_i + result_i] = result[2][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_min_maps[unit_i + result_i] = result[3][padding:padding+max_rf, padding:padding+max_rf]
                    # Record the number of bars used in each map (append to txt files).
                    record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                    result[4], result[5])
                    # record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                    #                 result[6], result[7])
                unit_i += real_batch_size

            # Save the maps of all units.
            weighte_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_max_sinemaps.npy")
            weighted_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_min_sinemaps.npy")
            # non_overlap_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_max_sinemaps.npy")
            # non_overlap_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_min_sinemaps.npy")
            np.save(weighte_max_maps_path, weighted_max_maps)
            np.save(weighted_min_maps_path, weighted_min_maps)
            # np.save(non_overlap_max_maps_path, non_overlap_max_maps)
            # np.save(non_overlap_min_maps_path, non_overlap_min_maps)

            # Make pdf for the layer.
            weighted_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_sinemaps.pdf")
            make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
            # non_overlap_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_sinemaps.pdf")
            # make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)

            end = time()

            total_time = end - start        
            
            speed_txt_path = os.path.join(result_dir, f"speed.txt")
            with open(speed_txt_path, 'a') as f:
                f.write(f"{num_stim} {total_time}\n")


##################################    PASU    #################################


def pasu_bw_run_01b_test_num_stim(model, model_name, result_dir, _debug=False, batch_size=10,
                                  num_shapes_list=[], conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4a.
    
    Parameters
    ----------
    model      - neural network.\n
    model_name - name of neural network. Used for txt file naming.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image sizes above are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_pasu_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_pasu_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_pasu_tb100.txt")
    
    # Delete previous files
    delete_all_npy_files(result_dir)
    if os.path.exists(tb1_path):
        os.remove(tb1_path)
    if os.path.exists(tb20_path):
        os.remove(tb20_path)
    if os.path.exists(tb100_path):
        os.remove(tb100_path)
    
    for conv_i in range(len(layer_indices)):
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue

        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = pasu_shape.stimset_dict_pasu_bw(xn, max_rf)
        
        if max_rf < 30:
            continue

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        # non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        # non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = pasu_shape.pasu_run_01b(splist, truncated_model,
                                          num_units, batch_size=batch_size,
                                          _debug=_debug)

        # Append to txt files that summarize the top and bottom bars.
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)
        
        # Save the splist in a text file.
        splist_path = os.path.join(result_dir, f"{layer_name}_splist.txt")
        if os.path.exists(splist_path):
            os.remove(splist_path)
        record_splist(splist_path, splist)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top{top_n}_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot{top_n}_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)

        
        for num_shapes in num_shapes_list:
            
            start = time()

            num_stim_result_dir = os.path.join(result_dir, str(num_stim))
            if not os.path.exists(num_stim_result_dir):
                os.makedirs(num_stim_result_dir)
                print(f"Result directory: {num_stim_result_dir}...")

            weighted_counts_path = os.path.join(result_dir, f"{model_name}_pasu_weighted_counts.txt")
            # non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_pasu_non_overlap_counts.txt")
            if os.path.exists(weighted_counts_path):
                os.remove(weighted_counts_path)
            # if os.path.exists(non_overlap_counts_path):
            #     os.remove(non_overlap_counts_path)

            # This block of code contained in the following while-loop used to be
            # a bottleneck of the program because it is all computed by a single
            # CPU core. Improvement by multiprocessing was implemented on August
            # 15, 2022 to solve the problem.
            batch_size = os.cpu_count() // 2
            unit_i = 0
            while (unit_i < num_units):
                if _debug and unit_i >= 20:
                    break
                real_batch_size = min(batch_size, num_units - unit_i)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(pasu_shape.make_shapemaps,
                                        [splist for _ in range(real_batch_size)],
                                        [center_responses for _ in range(real_batch_size)],
                                        [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                        [_debug for _ in range(real_batch_size)],
                                        [False for _ in range(real_batch_size)],
                                        [num_shapes for _ in range(real_batch_size)],
                                        )
                # Crop and save maps to layer-level array
                for result_i, result in enumerate(results):
                    weighted_max_maps[unit_i + result_i] = result[0][padding:padding+max_rf, padding:padding+max_rf]
                    weighted_min_maps[unit_i + result_i] = result[1][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_max_maps[unit_i + result_i] = result[2][padding:padding+max_rf, padding:padding+max_rf]
                    # non_overlap_min_maps[unit_i + result_i] = result[3][padding:padding+max_rf, padding:padding+max_rf]
                    # Record the number of bars used in each map (append to txt files).
                    record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                    result[4], result[5])
                    # record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                    #                 result[6], result[7])
                unit_i += real_batch_size

            # Save the maps of all units.
            weighte_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_max_shapemaps.npy")
            weighted_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_min_shapemaps.npy")
            # non_overlap_max_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_max_shapemaps.npy")
            # non_overlap_min_maps_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_min_shapemaps.npy")
            np.save(weighte_max_maps_path, weighted_max_maps)
            np.save(weighted_min_maps_path, weighted_min_maps)
            # np.save(non_overlap_max_maps_path, non_overlap_max_maps)
            # np.save(non_overlap_min_maps_path, non_overlap_min_maps)

            # Make pdf for the layer.
            weighted_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_weighted_shapemaps.pdf")
            make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
            # non_overlap_pdf_path = os.path.join(num_stim_result_dir, f"{layer_name}_non_overlap_shapemaps.pdf")
            # make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)
        
            end = time()

            total_time = end - start        
            
            speed_txt_path = os.path.join(result_dir, f"speed.txt")
            with open(speed_txt_path, 'a') as f:
                f.write(f"{num_shapes} {total_time}\n")
        
##############################################################################


if __name__ == '__main__':
    # Create the result directory if it does not exist.
    this_result_dir = os.path.join(result_dir, model_name, layer_name)
    if os.path.exists(result_dir) and not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
    
    start = time()
    if rfmp_name == 'rfmp4a':
        rfmp4a_run_01b_test_num_stim(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                        batch_size=batch_size, num_bars_list=num_stim_list, conv_i_to_run=conv_i_to_run,
                        response_thresholds=response_thresholds)
    elif rfmp_name == 'rfmp4c7o':
        rfmp4c7o_run_01_test_num_stim(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                        batch_size=batch_size, num_bars_list=num_stim_list, conv_i_to_run=conv_i_to_run)
    elif rfmp_name == 'rfmp_sin1':
        sin1_run_01b_test_num_stim(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                        batch_size=batch_size, num_stim_list=num_stim_list, conv_i_to_run=conv_i_to_run)
    elif rfmp_name == 'pasu':
        pasu_bw_run_01b_test_num_stim(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                        batch_size=batch_size, num_shapes_list=num_stim_list, conv_i_to_run=conv_i_to_run)
