""" 
Script to plot the trend of correlation coefficients of a model. The data it
uses is alrady in results/compare/map_correlations/{model_name}/.

Tony Fu, May 2, 2023
"""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c

# Specify the model and the Gaussian blur radius
model_name = 'alexnet'
sigma_rf_ratio = 1/30
max_or_min = 'max'

# Choose map names from: ['gt','gt_composite','occlude_composite','rfmp4a','rfmp4c7o']
map1_name = 'gt'
map2_name = 'rfmp4a'

# Load the data
file_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations', model_name)
data_file_path = os.path.join(file_dir, f"{max_or_min}_map_r_{sigma_rf_ratio:.4f}.txt")
data_df = pd.read_csv(data_file_path, sep=' ')

# Filter the DF to get only the columns I need
corr_df = data_df #TODO: get only the ['LAYER', 'UNIT', f"{map1_name}_vs_{map2_name}"] columns
layer_names = # TODO: get the uniques elements in 'LAYER' column, sorted such that ['conv1', 'conv2', ...]

# Plot the trend and save it
pdf_path = os.path.join(file_dir, f"{max_or_min}_{map1_name}_vs_{map2_name}_r_trend_{sigma_rf_ratio:.4f}.pdf")
with PdfPages(pdf_path) as pdf:
    for layer_name in layer_names:
        plt.subplot
        
