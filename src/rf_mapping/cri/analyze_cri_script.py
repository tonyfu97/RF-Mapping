"""
To visualize the difference between ground truth and bar mapping methods.

Tony Fu, Sep 20, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import pearsonr
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.result_txt_format import (CenterReponses as CR)
from src.rf_mapping.hook import ConvUnitCounter

# Please specify the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

is_random = False
this_is_a_test_run = True
top_n_r = 1


###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n]") 
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)
num_layers = len(rf_sizes)

# Load CRI
cri_num_images = 1000
cri_path = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'cri', model_name, f'cri_{cri_num_images}.txt')
cri_df = pd.read_csv(cri_path, sep=" ", header=None)
cri_df.columns = ['LAYER', 'UNIT', 'CRI']

# Load Rfmp4a dir
rfmp4a_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', model_name)

# Load Rfmp4c7o dir
rfmp4c7o_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'mapping', model_name)


pdf_path = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'cri', model_name, f"{top_n_r}_avg_cri.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(num_layers*5, 10))
    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"

        # Load Rfmp4a center responses
        top_rfmp4a_path = os.path.join(rfmp4a_dir, f'{layer_name}_top5000_responses.txt')
        top_rfmp4a_df = pd.read_csv(top_rfmp4a_path, sep=" ", header=None)
        top_rfmp4a_df.columns = [e.name for e in CR]

        # Load Rfmp4c7o center responses
        top_rfmp4c7o_path = os.path.join(rfmp4c7o_dir, f'{layer_name}_top5000_responses.txt')
        top_rfmp4c7o_df = pd.read_csv(top_rfmp4c7o_path, sep=" ", header=None)
        top_rfmp4c7o_df.columns = [e.name for e in CR]
        
        # Average the top-N resposnes for each unit
        top_rfmp4a_responses = top_rfmp4a_df.loc[(top_rfmp4a_df.RANK < top_n_r), ['UNIT', 'R']]
        avg_top_rfmp4a_responses = top_rfmp4a_responses.groupby('UNIT').mean()
        
        top_rfmp4c7o_responses = top_rfmp4c7o_df.loc[(top_rfmp4c7o_df.RANK < top_n_r), ['UNIT', 'R']]
        avg_top_rfmp4c7o_responses = top_rfmp4c7o_responses.groupby('UNIT').mean()
        
        plt.subplot(2, num_layers, conv_i+1)
        plt.scatter(avg_top_rfmp4a_responses, avg_top_rfmp4c7o_responses, alpha=0.4)
        plt.plot([-10, 100], [-10, 100], 'k', alpha=0.4)
        plt.xlim([-10, 100])
        plt.ylim([-10, 100])
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.xlabel(f"top-{top_n_r} achromatic", fontsize=14)
        plt.ylabel(f"top-{top_n_r} color", fontsize=14)
        plt.title(f"{layer_name}", fontsize=14)
        
        avg_top_responses_diff = np.array(avg_top_rfmp4c7o_responses - avg_top_rfmp4a_responses)[:,0]
        avg_top_responses_diff -= avg_top_responses_diff.min() - 1
        layer_cri = cri_df.loc[(cri_df.LAYER == layer_name), 'CRI'].to_numpy()
        
        # Log-transform the two axis before correlations.
        # avg_top_responses_diff = np.log(avg_top_responses_diff)
        # layer_cri = np.log(layer_cri)

        plt.subplot(2, num_layers, conv_i+1+num_layers)
        plt.scatter(avg_top_responses_diff, layer_cri, alpha=0.4)
        r_val, _ = pearsonr(avg_top_responses_diff, layer_cri)
        plt.title(f"r = {r_val:.4f}", fontsize=14)
        plt.xlabel("color - achromatic", fontsize=14)
        plt.ylabel("CRI", fontsize=14)
        
    pdf.savefig()
    plt.show()
    plt.close()
