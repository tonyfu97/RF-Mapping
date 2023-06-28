import os
import sys

from tqdm import tqdm
from torchvision import models

sys.path.append('../../..')
from src.rf_mapping.bar import *
import src.rf_mapping.constants as c
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.log import get_logger


# Please specify some details here:
MODEL = models.alexnet(pretrained=True).to(c.DEVICE)
MODEL_NAME = 'alexnet'
LAYER_NAME = 'conv3'
LAYER_INDEX = 6
MAX_RF = 99
XN = 127
NUM_UNITS = 384

RESULT_DIR = f"/Volumes/T7 Shield/borderownership/results (2023 summer)/rfmp4a/fnat/{MODEL_NAME}/{LAYER_NAME}"

################# Load the ground truth response of the layer #################

N_TOP_NATURAL = 10
gt_response_path = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', MODEL_NAME, f"{LAYER_NAME}_responses.npy")
gt_responses = np.load(gt_response_path)
gt_responses = np.sort(gt_responses, axis=1)
# Shape = [num_units, num_images, 2]. There are 2 columns:
# 0. Max responses of the given image and unit
# 1. Min responses of the given image and unit

# Averages the top- and bottom-N responses for all units
avg_top_gt_responses = np.mean(gt_responses[:, -N_TOP_NATURAL:, 0], axis=1)
avg_bot_gt_responses = np.mean(gt_responses[:, :N_TOP_NATURAL, 1], axis=1)

############################### Helper Functions ##############################


def get_splist(xn,
               max_rf,
               barlen_ratios = np.array([48/64, 24/64, 12/64, 6/64]),
               aspect_ratios = np.array([1/2, 1/5, 1/10]),
               orientations = np.arange(0.0, 180.0, 22.5),
               aa = 0.5,
               grid_divider = 2.0):
    splist = []
    barlen = barlen_ratios * max_rf
    for bl in barlen:
        xlist = stimset_gridx_map_with_divider(max_rf, bl, grid_divider)
        for ar in aspect_ratios:
            stim_dapp_bar_xyo_bw(splist, xn, xlist, orientations, bl, ar*bl, aa)
    return splist


def get_center_responses(splist, truncated_model, num_units, batch_size=100):
    return barmap_run_01b(splist, truncated_model, num_units, batch_size=batch_size)


def fnat(unit, top_bar_response, bot_bar_response):
    top_fnat = top_bar_response / avg_top_gt_responses[unit]
    bot_fnat = bot_bar_response / avg_bot_gt_responses[unit]
    return top_fnat, bot_fnat


def center_response_to_fnat(center_responses):
    top_fnats = np.zeros((NUM_UNITS))
    bot_fnats = np.zeros((NUM_UNITS))
    for unit in range(NUM_UNITS):
        isort = np.argsort(center_responses[:, unit])
        top_i = isort[-1]
        bot_i = isort[0]
        top_fnat, bot_fnat = fnat(unit, center_responses[top_i, unit], center_responses[bot_i, unit])
        top_fnats[unit] = top_fnat
        bot_fnats[unit] = bot_fnat
    return top_fnats, bot_fnats

##################################### Main ####################################


if __name__ == "__main__":
    truncated_model = get_truncated_model(MODEL, LAYER_INDEX)
    TEST_NAME = ""
    logger = get_logger(os.path.join(RESULT_DIR, f"fnat.log"), __file__)
    
    ############################# TEST BAR LENGTH #############################
    
    if TEST_NAME == "bar_length":  # drop each bar from default one by one
        barlen_ratios_list = [np.array([48/64, 24/64, 12/64]), # drop smallest
                              np.array([48/64, 24/64, 6/64]),  # drop medium-small
                              np.array([48/64, 12/64, 6/64]),  # drop medium-large
                              np.array([24/64, 12/64, 6/64]),  # drop largest
                              np.array([48/64, 24/64, 12/64, 6/64]),  # default
                              np.array([60/64, 48/64, 24/64, 12/64, 6/64]),  # add largest
                              np.array([48/64, 36/64, 24/64, 12/64, 6/64]),  # add medium-large
                              np.array([48/64, 24/64, 18/64, 12/64, 6/64]),  # add medium
                              np.array([48/64, 24/64, 12/64, 9/64,  6/64]),  # add medium-small
                              ]
        

        output_path = os.path.join(RESULT_DIR, f"{TEST_NAME}_fnat.txt")
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            f.write(f"barlen_ratio num_stim unit_id top_fnat bot_fnat\n")

        for barlen_i, barlen_ratios in enumerate(barlen_ratios_list):
            logger.info(f"barlen_ratios: {barlen_ratios}")
            splist = get_splist(XN, MAX_RF, barlen_ratios=barlen_ratios)
            center_responses = get_center_responses(splist, truncated_model, NUM_UNITS)
            top_fnat, bot_fnat = center_response_to_fnat(center_responses)

            with open(output_path, 'a') as f:
                for unit_id in tqdm(range(NUM_UNITS)):
                    f.write(f"{barlen_i} {len(splist)} {unit_id} {top_fnat[unit_id]:.4f} {bot_fnat[unit_id]:.4f}\n")

    ############################ TEST ASPECT RATIO ############################
    
    elif TEST_NAME == "aspect_ratio":
        aspect_ratios_list = [np.array([1/2, 1/5]),  # drop smallest
                              np.array([1/2, 1/10]), # drop medium
                              np.array([1/5, 1/10]), # drop largest
                              np.array([1/2, 1/5, 1/10]),  # default
                              np.array([1, 1/2, 1/5, 1/10]),  # add largest
                              np.array([1/2, 1/3, 1/5, 1/10]),  # add medium large
                              np.array([1/2, 1/5, 1/7, 1/10]),  # add medium small
                              np.array([1/2, 1/5, 1/10, 1/20]), # add smallest
                              ]

        output_path = os.path.join(RESULT_DIR, f"{TEST_NAME}_fnat.txt")
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            f.write(f"aspect_ratio num_stim unit_id top_fnat bot_fnat\n")

        for aspect_i, aspect_ratios in enumerate(aspect_ratios_list):
            logger.info(f"aspect_ratios: {aspect_ratios}")
            splist = get_splist(XN, MAX_RF, aspect_ratios=aspect_ratios)
            center_responses = get_center_responses(splist, truncated_model, NUM_UNITS)
            top_fnat, bot_fnat = center_response_to_fnat(center_responses)

            with open(output_path, 'a') as f:
                for unit_id in tqdm(range(NUM_UNITS)):
                    f.write(f"{aspect_i} {len(splist)} {unit_id} {top_fnat[unit_id]:.4f} {bot_fnat[unit_id]:.4f}\n")
    
    ############################ TEST ORIENTATIONS ############################
    
    elif TEST_NAME == "orientations":
        orientations_list = [np.arange(0.0, 180.0, 90),
                             np.arange(0.0, 180.0, 60),
                             np.arange(0.0, 180.0, 45),
                             np.arange(0.0, 180.0, 22.5),  # default
                             np.arange(0.0, 180.0, 22.5/2),
                             np.arange(0.0, 180.0, 22.5/4),]

        output_path = os.path.join(RESULT_DIR, f"{TEST_NAME}_fnat.txt")
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            f.write(f"orientation num_stim unit_id top_fnat bot_fnat\n")

        for ori_i, orientations in enumerate(orientations_list):
            logger.info(f"orientations: {orientations}")
            splist = get_splist(XN, MAX_RF, orientations=orientations)
            center_responses = get_center_responses(splist, truncated_model, NUM_UNITS)
            top_fnat, bot_fnat = center_response_to_fnat(center_responses)

            with open(output_path, 'a') as f:
                for unit_id in tqdm(range(NUM_UNITS)):
                    f.write(f"{ori_i} {len(splist)} {unit_id} {top_fnat[unit_id]:.4f} {bot_fnat[unit_id]:.4f}\n")
        
    ############################ TEST GRID DIVIDER ############################
    
    elif TEST_NAME == "grid_divider":
        grid_divider_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 2.0 is the default

        output_path = os.path.join(RESULT_DIR, f"{TEST_NAME}_fnat.txt")
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as f:
            f.write(f"orientation num_stim unit_id top_fnat bot_fnat\n")

        for gd_i, grid_divider in enumerate(grid_divider_list):
            logger.info(f"grid_divider: {grid_divider}")
            splist = get_splist(XN, MAX_RF, grid_divider=grid_divider)
            center_responses = get_center_responses(splist, truncated_model, NUM_UNITS)
            top_fnat, bot_fnat = center_response_to_fnat(center_responses)

            with open(output_path, 'a') as f:
                for unit_id in tqdm(range(NUM_UNITS)):
                    f.write(f"{gd_i} {len(splist)} {unit_id} {top_fnat[unit_id]:.4f} {bot_fnat[unit_id]:.4f}\n")
