"""
Plot the distrubtion of Gaussian fit fxvar (fraction of explained variance).

Tony Fu, Oct 13th, 2022
"""
import os
import sys
import math

import numpy as np
import pandas as pd
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import (GtGaussian as GT,
                                              Rfmp4aWeighted as W)


# Please specify some details here:
model_name = 'alexnet'
# model_name = 'vgg16'
# model_name = 'resnet18'
num_layers = 5


gaussian_map_dir = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit', model_name, 'abs')

gt_top_path = os.path.join(gaussian_map_dir, f'{model_name}_gt_gaussian_top.txt')
gt_bot_path = os.path.join(gaussian_map_dir, f'{model_name}_gt_gaussian_bot.txt')

gt_t_df = pd.read_csv(gt_top_path, sep=" ", header=None)
gt_b_df = pd.read_csv(gt_bot_path, sep=" ", header=None)

gt_t_df.columns = [e.name for e in GT]
gt_b_df.columns = [e.name for e in GT]

#######################################.#######################################

def make_fxvar_pdf():
    gt_fxvar = []
    gb_fxvar = []

    gt_labels = []
    gb_labels = []

    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"
        
        gt_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name), 'FXVAR']
        gt_data = gt_data[np.isfinite(gt_data)]
        gt_num_units = len(gt_data)
        gt_mean = gt_data.mean()
        gt_fxvar.append(gt_data)
        gt_labels.append(f"{layer_name}\n(n={gt_num_units},mu={gt_mean:.2f})")
        
        gb_data = gt_b_df.loc[(gt_b_df.LAYER == layer_name), 'FXVAR']
        gb_data = gb_data[np.isfinite(gb_data)]
        gb_num_units = len(gb_data)
        gb_mean = gb_data.mean()
        gb_fxvar.append(gb_data)
        gb_labels.append(f"{layer_name}\n(n={gb_num_units},mu={gb_mean:.2f})")


    pdf_path = os.path.join(gaussian_map_dir, f"{model_name}_fxvar_dist.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(num_layers*3, 7))
        # plt.suptitle(f"Fractions of explained variance ({model_name})", fontsize=24)

        plt.subplot(1,2,1)
        plt.grid()
        plt.boxplot(gt_fxvar, labels=gt_labels, showmeans=True)
        plt.ylabel('fxvar', fontsize=18)
        plt.title(f"{model_name} ground truth top", fontsize=18)

        plt.subplot(1,2,2)
        plt.grid()
        plt.boxplot(gb_fxvar, labels=gb_labels, showmeans=True)
        plt.title(f"{model_name} ground truth bottom", fontsize=18)

        pdf.savefig()
        plt.show()
        plt.close()
        
        return gt_fxvar

if __name__ == '__main__':
    gt_fxvar = make_fxvar_pdf()
    pass