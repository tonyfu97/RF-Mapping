"""
Test the windowed mapping algorithm on one unit at a time.

Update: This script has been refactored into 4a_windowed_mapping_test2.py.

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
import src.rf_mapping.constants as c

# Please specify some details here:
MODEL_NAME = 'alexnet'
LAYER_NAME = 'conv3'
LAYER_INDEX = 6
UNIT_IDX = 201
MAX_RF = 99
ALPHA = 4.0

pdf_path = f"{c.RESULTS_DIR}/rfmp4a_windowed/{MODEL_NAME}/tests/{LAYER_NAME}_windowed_bars_test.pdf"
with PdfPages(pdf_path) as pdf:
    for UNIT_IDX in tqdm(range(384)):
        if UNIT_IDX != 2: continue
        # Load the top bars of the unit
        top_bar_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_top5000_responses.txt")
        splist_path = os.path.join(c.RESULTS_DIR, 'rfmp4a', 'mapping', MODEL_NAME, f"{LAYER_NAME}_splist.txt")

        # Load the dataframes and name the columns
        top_bar_df = pd.read_csv(top_bar_path, sep='\s+')
        top_bar_df.columns = [e.name for e in CR]

        splist_df = pd.read_csv(splist_path, sep='\s+')
        splist_df.columns = [e.name for e in SPL]

        # Get the top-{rank} bar of the unit
        TOP_RANK = 0
        top_rank_bar_index = top_bar_df.loc[top_bar_df.UNIT == UNIT_IDX, 'STIM_I'].values[TOP_RANK].astype(int)
        top_rank_bar_response = top_bar_df.loc[top_bar_df.UNIT == UNIT_IDX, 'R'].values[TOP_RANK]

        # Get the stimuli list
        params = splist_df.loc[splist_df.STIM_I == top_rank_bar_index].to_dict('records')[0]

        # Get the model
        model = getattr(models, MODEL_NAME)(pretrained=True).to(c.DEVICE)
        truncated_model = get_truncated_model(model, LAYER_INDEX)

        # Make the bars
        bar = stimfr_bar(params['XN'], params['YN'],
                        params['X0'], params['Y0'],
                        params['THETA'], params['LEN'], params['WID'], 
                        params['AA'], params['FGVAL'], 0)
        # We will add the params['BGVAL'] later because we need to window the bar first

        def add_bgval_to_bar(bar, bgval):
            bar = bar * 2 + bgval
            return bar

        def gray_bar_to_rgb_tensor(bar):
            bar_rgb = np.repeat(bar[:, :, np.newaxis], 3, axis=2)
            bar_rgb = np.transpose(bar_rgb, (2, 0, 1))
            return torch.tensor(bar_rgb).type('torch.FloatTensor').to(c.DEVICE).unsqueeze(0)

        def get_center_response(bar_tensor):
            y = truncated_model(bar_tensor)
            yc, xc = calculate_center(y.shape[-2:])
            center_response = y[0, UNIT_IDX, yc, xc].detach().cpu().numpy()
            return center_response

        bar_w_bg = add_bgval_to_bar(bar, params['BGVAL'])
        bar_tensor = gray_bar_to_rgb_tensor(bar_w_bg)
        center_response = get_center_response(bar_tensor)
            
        # Make sure the responses are close to the one recorded in the text file
        assert math.isclose(top_rank_bar_response, center_response, abs_tol=0.01)

        # Window the bar
        edges_and_corners = get_coordinates_of_edges_and_corners(params['XN'], params['YN'],
                                                                params['X0'], params['Y0'],
                                                                -params['THETA'], params['LEN'], params['WID'])
        windowed_bars_responses = {}
        SIGMA = params['LEN'] / ALPHA
        THREHOLD = 1

        def get_windowed_bar(bar, coord):
            windowed_bar = bar * create_gaussian(SIGMA, coord, bar.shape)
            return windowed_bar
            

        plt.figure(figsize=(45, 5))
        plt.subplot(1, 9, 1)
        plt.imshow(bar_w_bg, cmap='gray', vmin=0, vmax=1) # interpolation='none', filternorm=False, rasterized=True)
        plt.title(f"Original bar, response={center_response:.2f}")
        plt.axis('off')

        for i, (name, coord) in enumerate(edges_and_corners.items()):
            windowed_bar = get_windowed_bar(bar, coord)
            windowed_bar_w_bg = add_bgval_to_bar(windowed_bar, params['BGVAL'])
            windowed_bar_tensor = gray_bar_to_rgb_tensor(windowed_bar_w_bg)
            windowed_bars_responses[name] = get_center_response(windowed_bar_tensor)
            
            plt.subplot(1, 9, i + 2)
            plt.imshow(windowed_bar_w_bg, cmap='gray', vmin=-1, vmax=1) # interpolation='none', filternorm=False, rasterized=True)
            plt.title(f"{name}, response={windowed_bars_responses[name]:.2f}")
            plt.axis('off')

        plt.suptitle(f"Unit {UNIT_IDX}", fontsize=16)
        pdf.savefig()
        # plt.show()
        plt.close()
