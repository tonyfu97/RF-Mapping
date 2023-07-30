"""
Test the windowed mapping algorithm on one unit at a time.

Explanation
-----------
Test3 improves on test2 by using the bars with original response greater than
0.5 of the maximum response (instead of just the top-1 bar). The allows us to
make a bar map. The 'map' is actually stored as a text file, where each row
represent a delta with the format:
    LAYER UNIT RANK_I STIM_I WINDOW R X Y

Tony Fu, June 2023
"""
import os
import sys
import math
import pandas as pd
from torchvision import models
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.bar import *
from src.rf_mapping.result_txt_format import CenterReponses as CR, Rfmp4aSplist as SPL
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.log import get_logger
from src.rf_mapping.model_utils import ModelInfo
import src.rf_mapping.constants as c

# Main script
# Please specify some details here:
MODEL_NAME = 'alexnet'
LAYER_NAME = 'conv'
ALPHA = 4.0  # window's sigma is bar_length / ALPHA. Default value is 4.0.
RESPONSE_THR = 0.5  # Only use bars with original response > max(r_max * RESPONSE_THR, 0)
MAX_NUM_BARS = 100

# Please specify the output directory and the pdf path
output_dir = f"{c.RESULTS_DIR}/rfmp4a/window/{MODEL_NAME}"
max_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_max_windowed_map.txt")
min_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_min_windowed_map.txt")

#################### DON'T TOUCH ANYTHING BELOW THIS LINE #####################

# Helper functions
def add_bgval_to_bar(bar, bgval):
    return bar * 2 + bgval

def gray_bar_to_rgb_tensor(bar):
    bar_rgb = np.repeat(bar[:, :, np.newaxis], 3, axis=2)
    bar_rgb = np.transpose(bar_rgb, (2, 0, 1))
    return torch.tensor(bar_rgb).type('torch.FloatTensor').to(c.DEVICE).unsqueeze(0)

def get_center_response(bar_tensor, truncated_model, unit_idx):
    y = truncated_model(bar_tensor)
    yc, xc = calculate_center(y.shape[-2:])
    return y[0, unit_idx, yc, xc].detach().cpu().numpy()

def get_windowed_bar(bar, coord, SIGMA):
    return bar * create_gaussian(SIGMA, coord, bar.shape)

def map_bar(rank_i, bar_i, rank, txt_path):
    # Get the stimuli list
    params = splist_df.loc[splist_df.STIM_I == bar_i].to_dict('records')[0]

    # Make the bars
    bar = stimfr_bar(params['XN'], params['YN'],
                    params['X0'], params['Y0'],
                    params['THETA'], params['LEN'], params['WID'], 
                    params['AA'], params['FGVAL'], 0)

    bar_w_bg = add_bgval_to_bar(bar, params['BGVAL'])
    bar_tensor = gray_bar_to_rgb_tensor(bar_w_bg)
    center_response = get_center_response(bar_tensor, truncated_model, unit_idx)
    
    # Make sure the responses are close to the one recorded in the text file
    assert math.isclose(rank, center_response, abs_tol=0.01)

    # Window the bar
    edges_and_corners = get_coordinates_of_edges_and_corners(params['XN'], params['YN'],
                                                            params['X0'], params['Y0'],
                                                            -params['THETA'], params['LEN'], params['WID'])
    windowed_bars_responses = {}
    SIGMA = params['LEN'] / ALPHA
    
    xc, yc = (params['XN']-1.0)/2.0, (params['YN']-1.0)/2.0

    with open(txt_path, 'a') as f:
        f.write(f"{LAYER_NAME} {unit_idx} {rank_i} {bar_i} original {center_response:.4f} {params['X0'] + xc} {params['Y0'] + yc}\n")

    for (name, coord) in edges_and_corners.items():
        windowed_bar = get_windowed_bar(bar, coord, SIGMA)
        windowed_bar_w_bg = add_bgval_to_bar(windowed_bar, params['BGVAL'])
        windowed_bar_tensor = gray_bar_to_rgb_tensor(windowed_bar_w_bg)
        windowed_bars_responses[name] = get_center_response(windowed_bar_tensor, truncated_model, unit_idx)
        
        # Record respnoses in the txt file and plot the bars
        with open(txt_path, 'a') as f:
            f.write(f"{LAYER_NAME} {unit_idx} {rank_i} {bar_i} {name} {windowed_bars_responses[name]:.4f} {coord[0]:.4f} {coord[1]:.4f}\n")

# Delete the txt files if it exists
if os.path.exists(max_txt_path):
    os.remove(max_txt_path)
if os.path.exists(min_txt_path):
    os.remove(min_txt_path)

# Write the header of the txt file
with open(max_txt_path, 'a') as f:
    f.write(f"LAYER UNIT RANK_I STIM_I WINDOW R X Y\n")
with open(min_txt_path, 'a') as f:
    f.write(f"LAYER UNIT RANK_I STIM_I WINDOW R X Y\n")

# Get model information
model_info = ModelInfo()
LAYER_INDEX = model_info.get_layer_index(MODEL_NAME, LAYER_NAME)
MAX_RF = model_info.get_rf_size(MODEL_NAME, LAYER_NAME)
XN = model_info.get_xn(MODEL_NAME, LAYER_NAME)
NUM_UNITS = model_info.get_num_units(MODEL_NAME, LAYER_NAME)

# Logging
logger = get_logger(os.path.join(output_dir, f"4a_windowed_mapping.log"), __file__)
logger.info(f"Model = {MODEL_NAME}, layer = {LAYER_NAME}, alpha = {ALPHA}, response_thr = {RESPONSE_THR}, max_num_bars = {MAX_NUM_BARS}")

# Load the bars of the unit
top_bar_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_top5000_responses.txt")
bot_bar_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_bot5000_responses.txt")
splist_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_splist.txt")

# Load the dataframes and name the columns
top_bar_df = pd.read_csv(top_bar_path, sep='\s+', header=None)
top_bar_df.columns = [e.name for e in CR]
bot_bar_df = pd.read_csv(bot_bar_path, sep='\s+', header=None)
bot_bar_df.columns = [e.name for e in CR]
splist_df = pd.read_csv(splist_path, sep='\s+', header=None)
splist_df.columns = [e.name for e in SPL]
WINDOWS = ['original'] + list(get_coordinates_of_edges_and_corners(0,0,0,0,0,1,1).keys())

# Get the model
model = getattr(models, MODEL_NAME)(pretrained=True).to(c.DEVICE)
truncated_model = get_truncated_model(model, LAYER_INDEX)
 
for unit_idx in tqdm(range(NUM_UNITS)):
    # Get the top-{rank} bar of the unit
    max_rank_bars = top_bar_df.loc[top_bar_df.UNIT == unit_idx, ['STIM_I', 'R']]
    min_rank_bars = bot_bar_df.loc[bot_bar_df.UNIT == unit_idx, ['STIM_I', 'R']]
    
    max_cutoff = max_rank_bars['R'].max() * RESPONSE_THR
    min_cutoff = min_rank_bars['R'].min() * RESPONSE_THR
    
    # Map the top/max bars
    for rank_i, (bar_i, r) in enumerate(zip(max_rank_bars['STIM_I'], max_rank_bars['R'])):
        if r > max_cutoff and rank_i < MAX_NUM_BARS:
            map_bar(rank_i, bar_i, r, max_txt_path)
        
    # Map the bottom/min bars
    for rank_i, (bar_i, r) in enumerate(zip(min_rank_bars['STIM_I'], min_rank_bars['R'])):
        if r < min_cutoff and rank_i < MAX_NUM_BARS:
            map_bar(rank_i, bar_i, r, min_txt_path)

logger.info(f"Done. Results saved to {max_txt_path} {min_txt_path}")
