"""
Test the windowed mapping algorithm on one layer at a time.

Plot the "maps" generated by `4a_windowed_mapping_multiple_bars.py` as a pdf.
The "maps" are actually a text file, and this script plots "max" the responses
of the windowed bars to the coordinates of the corners/edges of the windowed
bars.

Tony Fu, June 2023
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

# Please specify the source directory and the output pdf path
output_dir = f"/Volumes/T7 Shield/borderownership/results (2023 summer)/rfmp4a/window/{MODEL_NAME}"
max_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_max_windowed_map.txt")
min_txt_path = os.path.join(output_dir, f"{LAYER_NAME}_min_windowed_map.txt")
pdf_path = os.path.join(output_dir, f"{MODEL_NAME}_{LAYER_NAME}_windowed_bars.pdf")

# Log some information
logger = get_logger(os.path.join(output_dir, '4a_windowed_mapping.log'), __file__)
logger.info(f"MODEL_NAME: {MODEL_NAME}, LAYER_NAME: {LAYER_NAME}, SIGMA_TO_RF_RATIO: {SIGMA_TO_RF_RATIO}")

# Load the "maps"
max_map = pd.read_csv(max_txt_path, sep='\s+', header=0)
min_map = pd.read_csv(min_txt_path, sep='\s+', header=0)
# Format: LAYER UNIT RANK_I STIM_I WINDOW R X Y

# Get model information
model_info = ModelInfo()
LAYER_INDEX = model_info.get_layer_index(MODEL_NAME, LAYER_NAME)
MAX_RF = model_info.get_rf_size(MODEL_NAME, LAYER_NAME)
XN = model_info.get_xn(MODEL_NAME, LAYER_NAME)
NUM_UNITS = model_info.get_num_units(MODEL_NAME, LAYER_NAME)
SIGMA = MAX_RF / SIGMA_TO_RF_RATIO

# Plot delta maps
with PdfPages(pdf_path) as pdf:
    for unit_idx in tqdm(range(NUM_UNITS)):
        max_map_array = np.zeros((XN, XN))
        min_map_array = np.zeros((XN, XN))

        unit_max_map_df = max_map[max_map.UNIT == unit_idx]
        unit_min_map_df = min_map[min_map.UNIT == unit_idx]

        # Helper functions
        def make_idx(flt_idx):
            # clip and round
            idx = max(0, flt_idx)
            idx = min(XN-1, idx)
            return round(idx)

        for _, row in unit_max_map_df.iterrows():
            max_map_array[make_idx(row.Y), make_idx(row.X)] += max(row.R, 0)
            
        for _, row in unit_min_map_df.iterrows():
            min_map_array[make_idx(row.Y), make_idx(row.X)] += max(-row.R, 0)

        # Gaussian smooth the map
        max_map_array = gaussian_filter(max_map_array, sigma=SIGMA)
        min_map_array = gaussian_filter(min_map_array, sigma=SIGMA)
            
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{MODEL_NAME} {LAYER_NAME} unit {unit_idx}")
        plt.subplot(1, 2, 1)
        plt.imshow(max_map_array, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(min_map_array, cmap='gray')
        pdf.savefig()
        plt.close()
    
logger.info(f"Done. Results saved to {pdf_path}.")
