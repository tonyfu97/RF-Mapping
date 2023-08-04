"""
Test the windowed mapping algorithm on one layer at a time.

Plot the "maps" generated by `4a_windowed_mapping_multiple_bars.py` as a pdf.
Different from `4a_windowed_mapping_multiple_bars_pdf.py`, this script plots
the windowed bars weighted by the responses, rather than just the coordinates
("delta") of the corners/edges.

Tony Fu, July 2023
"""
import os
import sys
import math
import pandas as pd
from torchvision import models
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
LAYER_NAME = 'conv5'
SIGMA_TO_RF_RATIO = 20
ALPHA = 4.0  # window's sigma is bar_length / ALPHA. Default value is 4.0.

# Please specify the source directory and the output pdf path
output_dir = f"{c.RESULTS_DIR}/rfmp4a_windowed/mapping/{MODEL_NAME}"
max_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_max_windowed_map.txt")
min_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_min_windowed_map.txt")
splist_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_splist.txt")
pdf_path = os.path.join(output_dir, f"{MODEL_NAME}_{LAYER_NAME}_windowed_barmaps.pdf")
max_npy_output_path = os.path.join(output_dir, f"{LAYER_NAME}_windowed_bars_max.npy")
min_npy_output_path = os.path.join(output_dir, f"{LAYER_NAME}_windowed_bars_min.npy")

# Log some information
logger = get_logger(os.path.join(output_dir, '4a_windowed_mapping.log'), __file__)
logger.info(f"MODEL_NAME: {MODEL_NAME}, LAYER_NAME: {LAYER_NAME}, SIGMA_TO_RF_RATIO: {SIGMA_TO_RF_RATIO}")

# Load the "maps"
max_map = pd.read_csv(max_txt_path, sep='\s+', header=0)
min_map = pd.read_csv(min_txt_path, sep='\s+', header=0)
# Format: LAYER UNIT RANK_I STIM_I WINDOW R X Y

# Load stimulus parameters
splist_df = pd.read_csv(splist_path, sep='\s+', header=None)
splist_df.columns = [e.name for e in SPL]

# Get model information
model_info = ModelInfo()
LAYER_INDEX = model_info.get_layer_index(MODEL_NAME, LAYER_NAME)
MAX_RF = model_info.get_rf_size(MODEL_NAME, LAYER_NAME)
XN = model_info.get_xn(MODEL_NAME, LAYER_NAME)
NUM_UNITS = model_info.get_num_units(MODEL_NAME, LAYER_NAME)
SIGMA = MAX_RF / SIGMA_TO_RF_RATIO

#################### DON'T TOUCH ANYTHING BELOW THIS LINE #####################

# Helper functions
def _get_windowed_bar(bar, coord, SIGMA):
    return bar * create_gaussian(SIGMA, coord, bar.shape)

def get_windowed_bar(bar_i, window):
    # Get the stimuli list
    params = splist_df.loc[splist_df.STIM_I == bar_i].to_dict('records')[0]

    # Make the bars
    bar = stimfr_bar(params['XN'], params['YN'],
                    params['X0'], params['Y0'],
                    params['THETA'], params['LEN'], params['WID'], 
                    params['AA'], 1, 0)
    
    if window == 'original':
        return bar
    
    # Window the bar
    edges_and_corners = get_coordinates_of_edges_and_corners(params['XN'], params['YN'],
                                                            params['X0'], params['Y0'],
                                                            -params['THETA'], params['LEN'], params['WID'])
    SIGMA = params['LEN'] / ALPHA
    coord = edges_and_corners[window]
    windowed_bar = _get_windowed_bar(bar, coord, SIGMA)
    return windowed_bar

def crop_map(map_array, xn=XN, rf_size=MAX_RF):
    # Crop the map
    map_array = map_array[(xn - rf_size) // 2 : (xn + rf_size) // 2,
                          (xn - rf_size) // 2 : (xn + rf_size) // 2]
    return map_array

###############################################################################

# Plot maps
with PdfPages(pdf_path) as pdf:
    max_maps_array = np.zeros((NUM_UNITS, MAX_RF, MAX_RF))
    min_maps_array = np.zeros((NUM_UNITS, MAX_RF, MAX_RF))
    for unit_idx in tqdm(range(NUM_UNITS)):
        max_map_array = np.zeros((MAX_RF, MAX_RF))
        min_map_array = np.zeros((MAX_RF, MAX_RF))

        unit_max_map_df = max_map[max_map.UNIT == unit_idx]
        unit_min_map_df = min_map[min_map.UNIT == unit_idx]
        
        bar_i_max_list = unit_max_map_df.STIM_I.unique()
        bar_i_min_list = unit_min_map_df.STIM_I.unique()
        
        for bar_i in bar_i_max_list:
            # find the WINDOW with the largest R for this STIM_I
            unit_max_map_for_stim_i = unit_max_map_df[unit_max_map_df.STIM_I == bar_i]
            unit_max_map_for_stim_i = unit_max_map_for_stim_i[unit_max_map_for_stim_i.WINDOW != 'original']
            unit_max_map_for_stim_i = unit_max_map_for_stim_i.sort_values(by='R', ascending=False)
            max_window = unit_max_map_for_stim_i.iloc[0].WINDOW
            R = unit_max_map_for_stim_i.iloc[0].R

            if R > 0:
                max_map_array += crop_map(get_windowed_bar(bar_i, max_window)) * R

        for bar_i in bar_i_min_list:
            # find the WINDOW with the smallest for this STIM_I
            unit_min_map_for_stim_i = unit_min_map_df[unit_min_map_df.STIM_I == bar_i]
            unit_min_map_for_stim_i = unit_min_map_for_stim_i[unit_min_map_for_stim_i.WINDOW != 'original']
            unit_min_map_for_stim_i = unit_min_map_for_stim_i.sort_values(by='R', ascending=True)
            min_window = unit_min_map_for_stim_i.iloc[0].WINDOW
            R = unit_min_map_for_stim_i.iloc[0].R
            
            if R < 0:
                min_map_array += crop_map(get_windowed_bar(bar_i, min_window)) * abs(R)
        
        # Save the maps to array
        max_maps_array[unit_idx] = max_map_array
        min_maps_array[unit_idx] = min_map_array
    
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{MODEL_NAME} {LAYER_NAME} unit {unit_idx}")
        plt.subplot(1, 2, 1)
        plt.imshow(max_map_array, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(min_map_array, cmap='gray')
        pdf.savefig()
        plt.close()
    
# Save the maps to npy
np.save(max_npy_output_path, max_maps_array)
np.save(min_npy_output_path, min_maps_array)
    
logger.info(f"Done. Results saved to {pdf_path}, {max_npy_output_path}, {min_npy_output_path}")