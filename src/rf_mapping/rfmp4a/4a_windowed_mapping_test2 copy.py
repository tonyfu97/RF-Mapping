"""
Test the windowed mapping algorithm on one unit at a time.

Explanation
-----------
We windowed the top bar of each unit and get the responses of the windowed bars.
The bar (windowed or not) and the responses are plotted in a pdf file. This
is just a refactor of 4a_windowed_mapping_test.py. This is no functional
difference between the two scripts, except that this script runs 3x faster.

Tony Fu, June 2023
"""
import os
import sys
import math
import pandas as pd
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
LAYER_NAME = 'conv1'
TOP_RANK = 0   # If 0, then the top bar of the unit will be used, etc.
ALPHA = 4.0

# Please specify the output directory and the pdf path
output_dir = f"{c.RESULTS_DIR}/rfmp4a/window/{MODEL_NAME}/tests"
pdf_path = os.path.join(output_dir, f"{LAYER_NAME}_windowed_bars_test2.pdf")
txt_path = os.path.join(output_dir, f"{LAYER_NAME}_windowed_bars_test2.txt")


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

def plot_bar(bar, name, response, idx):
    plt.subplot(1, 9, idx)
    plt.imshow(bar, cmap='gray', vmin=-1, vmax=1) 
    plt.title(f"{name}, response={response:.2f}")
    plt.axis('off')

# Delete the txt file if it exists
if os.path.exists(txt_path):
    os.remove(txt_path)

# Write the header of the txt file
with open(txt_path, 'a') as f:
    f.write(f"LAYER UNIT original ")
    f.write(f"{' '.join(list(get_coordinates_of_edges_and_corners(0,0,0,0,0,1,1).keys()))}\n")

# Get model information
model_info = ModelInfo()
LAYER_INDEX = model_info.get_layer_index(MODEL_NAME, LAYER_NAME)
MAX_RF = model_info.get_rf_size(MODEL_NAME, LAYER_NAME)
XN = model_info.get_xn(MODEL_NAME, LAYER_NAME)
NUM_UNITS = model_info.get_num_units(MODEL_NAME, LAYER_NAME)

# Logging
logger = get_logger(os.path.join(output_dir, f"4a_windowed_bars_test2.log"), __file__)
logger.info(f"Model = {MODEL_NAME}, layer = {LAYER_NAME}, top rank = {TOP_RANK}, alpha = {ALPHA}")

# Load the top bars of the unit
top_bar_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_top5000_responses.txt")
splist_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_splist.txt")

# Load the dataframes and name the columns
top_bar_df = pd.read_csv(top_bar_path, sep='\s+')
top_bar_df.columns = [e.name for e in CR]
splist_df = pd.read_csv(splist_path, sep='\s+')
splist_df.columns = [e.name for e in SPL]

# Get the model
model = getattr(models, MODEL_NAME)(pretrained=True).to(c.DEVICE)
truncated_model = get_truncated_model(model, LAYER_INDEX)

with PdfPages(pdf_path) as pdf:    
    for unit_idx in tqdm(range(NUM_UNITS)):
        # Get the top-{rank} bar of the unit
        top_rank_bar_index = top_bar_df.loc[top_bar_df.UNIT == unit_idx, 'STIM_I'].values[TOP_RANK].astype(int)
        top_rank_bar_response = top_bar_df.loc[top_bar_df.UNIT == unit_idx, 'R'].values[TOP_RANK]

        # Get the stimuli list
        params = splist_df.loc[splist_df.STIM_I == top_rank_bar_index].to_dict('records')[0]

        # Make the bars
        bar = stimfr_bar(params['XN'], params['YN'],
                        params['X0'], params['Y0'],
                        params['THETA'], params['LEN'], params['WID'], 
                        params['AA'], params['FGVAL'], 0)

        bar_w_bg = add_bgval_to_bar(bar, params['BGVAL'])
        bar_tensor = gray_bar_to_rgb_tensor(bar_w_bg)
        center_response = get_center_response(bar_tensor, truncated_model, unit_idx)
            
        # Make sure the responses are close to the one recorded in the text file
        assert math.isclose(top_rank_bar_response, center_response, abs_tol=0.01)

        # Window the bar
        edges_and_corners = get_coordinates_of_edges_and_corners(params['XN'], params['YN'],
                                                                params['X0'], params['Y0'],
                                                                -params['THETA'], params['LEN'], params['WID'])
        windowed_bars_responses = {}
        SIGMA = params['LEN'] / ALPHA

        plt.figure(figsize=(45, 5))
        plot_bar(bar_w_bg, 'Original bar', center_response, 1)
        with open(txt_path, 'a') as f:
            f.write(f"{LAYER_NAME} {unit_idx} {center_response:.4f}")

        for i, (name, coord) in enumerate(edges_and_corners.items()):
            windowed_bar = get_windowed_bar(bar, coord, SIGMA)
            windowed_bar_w_bg = add_bgval_to_bar(windowed_bar, params['BGVAL'])
            windowed_bar_tensor = gray_bar_to_rgb_tensor(windowed_bar_w_bg)
            windowed_bars_responses[name] = get_center_response(windowed_bar_tensor, truncated_model, unit_idx)
            
            # Record respnoses in the txt file and plot the bars
            with open(txt_path, 'a') as f:
                f.write(f" {windowed_bars_responses[name]:.2f}")
            plot_bar(windowed_bar_w_bg, name, windowed_bars_responses[name], i+2)
        
        with open(txt_path, 'a') as f:
            f.write('\n')

        plt.suptitle(f"Unit {unit_idx}", fontsize=16)
        pdf.savefig()
        plt.close()

logger.info(f"Done. Results saved to {pdf_path} and {txt_path}")
