"""
Code to plot the distribution of the RFMP statistics.

Tony Fu, Sep, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.spatial import get_num_layers
from src.rf_mapping.result_txt_format import CenterReponses as CR


# Please confirm the model
model = models.alexnet()
model_name = 'alexnet'

# Please confirm the directories
source_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', model_name)
result_dir = source_dir

# Please specify the what maps to compare:
# Choices are: 'gt', 'occlude', 'rfmp4a', and 'rfmp4c7o'.
map1_name = 'gt'
map2_name = 'rfmp4a'

# Result paths:
def load_maps(map_name, layer_name, max_or_min):
    """Loads the maps of the layer."""
    mapping_dir = os.path.join(c.REPO_DIR, 'results')
    
    if map_name == 'gt':
        mapping_path = os.path.join(mapping_dir,
                                    'ground_truth',
                                    'backprop_sum',
                                    model_name,
                                    'abs',
                                    f"{layer_name}_{max_or_min}.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'occlude':
        mapping_path = os.path.join(mapping_dir,
                                    'occlude',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_{max_or_min}.npy")
        maps = np.load(mapping_path)  # [top_n, unit, y, x]
        return np.mean(maps, axis=0)
    elif map_name == 'rfmp4a':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4a',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        return np.load(mapping_path)  # [unit, yn, xn]
    elif map_name == 'rfmp4c7o':
        mapping_path = os.path.join(mapping_dir,
                                    'rfmp4c7o',
                                    'mapping',
                                    model_name,
                                    f"{layer_name}_weighted_{max_or_min}_barmaps.npy")
        maps = np.load(mapping_path)  # [unit, 3, yn, xn]
        return np.mean(maps, axis=1)
    else:
        raise KeyError(f"{map_name} does not exist.")


# Load the statistics of the top and bottom 5000 bars.
num_layers = get_num_layers(model)

# Define helper functions
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]

def normalize_responses(responses):
    if not isinstance(responses, np.ndarray):
        responses = np.array(responses)
    r_max = responses.max()
    r_min = responses.min()
    r_range = r_max - r_min
    if r_range != 0:
        return (responses - r_min) / r_range
    else:
        return responses - r_min

for conv_i in range(num_layers):
    # if conv_i != 2: continue
    layer_name = f"conv{conv_i + 1}"
    top_file_path = os.path.join(source_dir, f'{layer_name}_top5000_responses.txt')
    bot_file_path = os.path.join(source_dir, f'{layer_name}_bot5000_responses.txt')
    top_r_df = pd.read_csv(top_file_path, sep=' ', header=None)
    bot_r_df = pd.read_csv(bot_file_path, sep=' ', header=None)

    # Name the columns.
    set_column_names(top_r_df, CR)
    set_column_names(bot_r_df, CR)
    
    num_units = len(top_r_df.UNIT.unique())
    
    maps1 = load_maps(map1_name, layer_name, 'max')
    maps2 = load_maps(map2_name, layer_name, 'max')
    
    r_vals = []
    intergrals = []
    
    for unit_i in range(num_units):
        # if unit_i != 13: continue
        r_val, p_val = pearsonr(maps1[unit_i].flatten(), maps2[unit_i].flatten())
        r_vals.append(r_val)
        # plt.figure(figsize=(10,3))
        responses = top_r_df.loc[(top_r_df.UNIT == unit_i), 'R']
        responses = normalize_responses(responses)
        intergrals.append(np.sum(responses))
        # plt.subplot(1,3,1)
        # plt.plot(np.arange(1, 5001), responses)
        # plt.title(f"{layer_name} no.{unit_i} top 5000 responses (r = {r_val:.2f})")
        # plt.xlabel('ranking')
        # plt.ylabel('response')
        
        # plt.subplot(1,3,2)
        # plt.imshow(maps1[unit_i])
        
        # plt.subplot(1,3,3)
        # plt.imshow(maps2[unit_i])
        # plt.show()
        
        # responses = bot_r_df.loc[(bot_r_df.UNIT == unit_i), 'R']
        # responses = normalize_responses(responses) + 1
        # plt.subplot(1,2,2)
        # plt.plot(np.arange(1, 5001), 1/responses)
        # plt.title(f"{layer_name} no.{unit_i} bottom 5000 responses")
        # plt.xlabel('ranking')
        # plt.ylabel('response')
        # plt.show()
    plt.scatter(r_vals, intergrals)
    plt.xlabel('r (gt and rfmp4a correlation)')
    plt.ylabel('area under sorted response curve')
    plt.title(f"{layer_name}")
    plt.show()
        
        
