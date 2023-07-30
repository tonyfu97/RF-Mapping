"""
To visualize the difference between guided-backprop and occlusion maps.

Tony Fu, Sep 27, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import RfmpHotSpot as HS

# Please specify the model
num_layers_dict = {'alexnet' : 5 , 'vgg16': 13, 'resnet18': 20}


#########################  LOAD MAP CORRELATIONS  #############################

map_corr_dict = {}
for model_name in num_layers_dict.keys():
    max_map_corr_path = os.path.join(c.RESULTS_DIR, 'compare', 'map_correlations',
                                    model_name, f"max_map_r_0.0333.txt")

    max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)

    map_corr_dict[model_name] =\
            max_map_corr_df[['LAYER', 'UNIT', 'gt_composite_vs_occlude_composite']].copy()

###########################  LOAD HOT SPOT DATA  ##############################

hot_spot_err_dist_dict = {}
for model_name in num_layers_dict.keys():
    gt_df_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit',
                              model_name, 'abs', f"{model_name}_gt_composite_hot_spot.txt")
    occlude_df_path = os.path.join(c.RESULTS_DIR, 'occlude', 'gaussian_fit',
                              model_name, f"{model_name}_occlude_composite_hot_spot.txt")

    gt_df = pd.read_csv(gt_df_path, sep=" ", header=0)
    occlude_df = pd.read_csv(occlude_df_path, sep=" ", header=0)

    gt_df.columns = [e.name for e in HS]
    occlude_df.columns = [e.name for e in HS]

    err_dist_df = gt_df[['LAYER', 'UNIT']].copy()
    err_dist_df['ERR_DIST'] = np.sqrt(
                                np.square(gt_df.loc[:,'TOP_X'] - occlude_df.loc[:,'TOP_X']) + 
                                np.square(gt_df.loc[:,'TOP_Y'] - occlude_df.loc[:,'TOP_Y']))
    
    hot_spot_err_dist_dict[model_name] = err_dist_df

################################  MAKE PDF  ###################################

pdf_path = os.path.join(c.RESULTS_DIR, 'compare',
                        f"gt_composite_vs_occlude_composite.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_corr_data = []
        for conv_i in range(num_layers):
            if conv_i == 0: continue
            df = map_corr_dict[model_name]
            corr_mean = df.loc[df.LAYER == f'conv{conv_i+1}', 'gt_composite_vs_occlude_composite'].to_numpy().mean()
            avg_corr_data.append(corr_mean)

        plt.plot(avg_corr_data, '.-', markersize=10, alpha=0.8)
        plt.ylim([0.5, 1.1])
        plt.ylabel('Direct map correlation', fontsize=16)
        plt.title(f"Comparing ground truths: Guided backprop composite vs. occlusion composite",
                    fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xlabel('conv layer index', fontsize=16)
    plt.xticks(np.arange(0,max_num_layers-1,2),  np.arange(2, max_num_layers+1,2), fontsize=16)
    plt.legend(num_layers_dict.keys(), fontsize=16)
    pdf.savefig()
    plt.show()
    plt.close()


    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_err_dist_data = []
        for conv_i in range(num_layers):
            if conv_i == 0: continue
            df = hot_spot_err_dist_dict[model_name]
            corr_mean = df.loc[df.LAYER == f'conv{conv_i+1}', 'ERR_DIST'].to_numpy().mean()
            avg_err_dist_data.append(corr_mean)

        plt.plot(avg_err_dist_data, '.-', markersize=20)
        plt.ylim([0, 10])
        plt.ylabel('Avg hot spot error distance (pix)', fontsize=16)
        plt.title(f"Comparing ground truths: Guided backprop composite vs. occlusion composite",
                    fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xlabel('conv layer index', fontsize=16)
    plt.xticks(np.arange(0,max_num_layers-1,2),  np.arange(2, num_layers+1,2), fontsize=16)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()
