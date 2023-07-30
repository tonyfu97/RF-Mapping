"""
Make a plot of RF size of each the layers of all models.

Tony Fu, May 3, 2023
"""
import os
import sys

import numpy as np
import pandas as pd
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.model_utils import ModelInfo

# Please specify some details here:
MODEL_NAMES = ['alexnet', 'vgg16', 'resnet18']
RESULT_DIR = os.path.join(c.RESULTS_DIR)
NUM_MAX_LAYERS = 20

MODEL_INFO_FILE_PATH = os.path.join(c.REPO_DIR, "data", "model_info.txt")
model_info = pd.read_csv(MODEL_INFO_FILE_PATH, delim_whitespace=True)
model_info.columns = ["model", "layer", "layer_index", "rf_size", "xn", "num_units"]

pdf_path = os.path.join(RESULT_DIR, f"rf_size.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(10,5))
    for model_name in MODEL_NAMES:
        rf_sizes = model_info.loc[(model_info['model'] == model_name), 'rf_size'].to_numpy()
        plt.plot(rf_sizes, '.-', markersize=16, label=model_name)

    plt.xticks(np.arange(NUM_MAX_LAYERS), np.arange(1, NUM_MAX_LAYERS+1), fontsize=16)
    plt.hlines(227, xmin=-1, xmax=20, label='image size', colors=['black'])
    plt.xlim([-1, 20])
    plt.grid(which='major')
    plt.legend(fontsize=16)
    plt.xlabel('conv layer number', fontsize=20)
    plt.ylabel('MRF size', fontsize=20)

    pdf.savefig()
    plt.close
