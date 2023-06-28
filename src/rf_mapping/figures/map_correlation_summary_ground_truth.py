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
from src.rf_mapping.result_txt_format import (RfmpHotSpot as HS,
                                              GtGaussian as GTG)

# Please specify the model
num_layers_dict = {'alexnet' : 5} #, 'vgg16': 13, 'resnet18': 20}

this_is_a_test_run = False
map1_name = 'gt'         # ['gt', 'gt_composite', 'occlude_composite', 'gradient_ascent']
map2_name = 'gradient_ascent'
fxvar_thres = 0.7
sigma_rf_ratio = 1/30

#########################  LOAD MAP CORRELATIONS  #############################

map_corr_dict = {}
for model_name in num_layers_dict.keys():
    max_map_corr_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'map_correlations',
                                     model_name, 'ground_truth', f"max_map_r_{sigma_rf_ratio:.4f}.txt")

    max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)

    map_corr_dict[model_name] =\
            max_map_corr_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']].copy()


########################  LOAD INTERSECTION OVER UNION  #######################

# map_iou_dict = {}
# for model_name in num_layers_dict.keys():
#     max_map_iou_path = os.path.join(c.REPO_DIR, 'results', 'compare', 'iou',
#                                     model_name, f"max_map_iou_{sigma_rf_ratio:.4f}.txt")

#     max_map_iou_df = pd.read_csv(max_map_iou_path, sep=" ", header=0)

#     map_iou_dict[model_name] =\
#             max_map_iou_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']].copy()


################################  MAKE PDF  ###################################

pdf_path = os.path.join(c.REPO_DIR, 'results', 'compare', f'{map1_name}_vs_{map2_name}',
                        f"{map1_name}_vs_{map2_name}_summary.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_corr_data = []
        all_corr_data = []
        for conv_i in range(num_layers):
            # if conv_i == 0: continue
            df = map_corr_dict[model_name]
            corr_data = df.loc[df.LAYER == f'conv{conv_i+1}', f'{map1_name}_vs_{map2_name}'].dropna().to_numpy()
            corr_mean = corr_data.mean()
            all_corr_data.append(corr_data)
            avg_corr_data.append(corr_mean)

        plt.plot(avg_corr_data, '.-', markersize=20)
        # Uncomment plt.boxplot() to visualize the spread. not recommended if plotting multiple models
        # plt.boxplot(all_corr_data, positions=range(len(all_corr_data)), widths=0.6)
        # plt.ylim([0.0, 1.1])
        plt.ylabel('Correlation Coefficient', fontsize=16)
        # plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    plt.xlabel('conv layer number', fontsize=20)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()
  
  
    # plt.figure(figsize=(12, 6))
    # for model_name, num_layers in num_layers_dict.items():
    #     avg_iou_data = []
    #     all_iou_data = []
    #     for conv_i in range(num_layers):
    #         # if conv_i == 0: continue
    #         df = map_iou_dict[model_name]
    #         iou_data = df.loc[df.LAYER == f'conv{conv_i+1}', f'{map1_name}_vs_{map2_name}'].dropna().to_numpy()
    #         iou_mean = iou_data.mean()
    #         all_iou_data.append(iou_data)
    #         avg_iou_data.append(iou_mean)

    #     plt.plot(avg_iou_data, '.-', markersize=20)
    #     # Uncomment plt.boxplot() to visualize the spread. not recommended if plotting multiple models
    #     # plt.boxplot(all_iou_data, positions=range(len(all_iou_data)), widths=0.6)
    #     plt.ylim([0.0, 0.7])
    #     plt.ylabel('IOU', fontsize=16)
    #     # plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    # max_num_layers = max(num_layers_dict.values())
    # plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    # plt.xlabel('conv layer number', fontsize=20)
    # plt.legend(num_layers_dict.keys(), fontsize=16)

    # pdf.savefig()
    # plt.show()
    # plt.close()

