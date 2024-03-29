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
num_layers_dict = {'alexnet' : 5, 'vgg16': 13, 'resnet18': 20}

this_is_a_test_run = False
map1_name = 'gt'         # ['gt', 'occlude']
map2_name = 'rfmp4c7o'   # ['rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']
fxvar_thres = 0.7
sigma_rf_ratio = 1/30

#########################  LOAD MAP CORRELATIONS  #############################

map_corr_dict = {}
for model_name in num_layers_dict.keys():
    max_map_corr_path = os.path.join(c.RESULTS_DIR, 'compare', 'map_correlations',
                                    model_name, f"max_map_r_{sigma_rf_ratio:.4f}.txt")

    max_map_corr_df = pd.read_csv(max_map_corr_path, sep=" ", header=0)

    map_corr_dict[model_name] =\
            max_map_corr_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']].copy()


########################  LOAD INTERSECTION OVER UNION  #######################

map_iou_dict = {}
for model_name in num_layers_dict.keys():
    max_map_iou_path = os.path.join(c.RESULTS_DIR, 'compare', 'iou',
                                    model_name, f"max_map_iou_{sigma_rf_ratio:.4f}.txt")

    max_map_iou_df = pd.read_csv(max_map_iou_path, sep=" ", header=0)

    map_iou_dict[model_name] =\
            max_map_iou_df[['LAYER', 'UNIT', f'{map1_name}_vs_{map2_name}']].copy()


###########################  LOAD HOT SPOT DATA  ##############################

err_dist_dict = {}
for model_name in num_layers_dict.keys():
    gt_df_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit',
                              model_name, 'abs', f"{model_name}_{map1_name}_hot_spot.txt")
    rfmp_df_path = os.path.join(c.RESULTS_DIR, map2_name, 'gaussian_fit',
                              model_name, f"{model_name}_{map2_name}_hot_spot.txt")

    gt_df = pd.read_csv(gt_df_path, sep=" ", header=0)
    rfmp_df = pd.read_csv(rfmp_df_path, sep=" ", header=0)

    gt_df.columns = [e.name for e in HS]
    rfmp_df.columns = [e.name for e in HS]

    err_dist_df = gt_df[['LAYER', 'UNIT']].copy()
    err_dist_df['HOT_SPOT_ERR_DIST'] = np.sqrt(
                                np.square(gt_df.loc[:,'TOP_X'] - rfmp_df.loc[:,'TOP_X']) + 
                                np.square(gt_df.loc[:,'TOP_Y'] - rfmp_df.loc[:,'TOP_Y']))
    
    err_dist_dict[model_name] = err_dist_df


###########################  LOAD HOT SPOT DATA  ##############################

for model_name, err_dist_df in err_dist_dict.items():
    gt_df_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'gaussian_fit',
                              model_name, 'abs', f"{model_name}_{map1_name}_gaussian_top.txt")
    rfmp_df_path = os.path.join(c.RESULTS_DIR, map2_name, 'gaussian_fit',
                              model_name, f"{model_name}_{map2_name}_gaussian_top.txt")

    gt_df = pd.read_csv(gt_df_path, sep=" ", header=0)
    rfmp_df = pd.read_csv(rfmp_df_path, sep=" ", header=0)

    gt_df.columns = [e.name for e in GTG]
    rfmp_df.columns = [e.name for e in GTG]

    err_dist_df['GAUSSIAN_ERR_DIST'] = np.sqrt(
                                np.square(gt_df.loc[:,'MUX'] - rfmp_df.loc[:,'MUX']) + 
                                np.square(gt_df.loc[:,'MUY'] - rfmp_df.loc[:,'MUY']))


################################  MAKE PDF  ###################################

pdf_path = os.path.join(c.RESULTS_DIR, 'compare', f'{map1_name}_vs_{map2_name}',
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
        plt.ylim([0.0, 1.1])
        plt.ylabel('Correlation Coefficient', fontsize=16)
        # plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    plt.xlabel('conv layer number', fontsize=20)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()
  
  
    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_iou_data = []
        all_iou_data = []
        for conv_i in range(num_layers):
            # if conv_i == 0: continue
            df = map_iou_dict[model_name]
            iou_data = df.loc[df.LAYER == f'conv{conv_i+1}', f'{map1_name}_vs_{map2_name}'].dropna().to_numpy()
            iou_mean = iou_data.mean()
            all_iou_data.append(iou_data)
            avg_iou_data.append(iou_mean)

        plt.plot(avg_iou_data, '.-', markersize=20)
        # Uncomment plt.boxplot() to visualize the spread. not recommended if plotting multiple models
        # plt.boxplot(all_iou_data, positions=range(len(all_iou_data)), widths=0.6)
        plt.ylim([0.0, 0.7])
        plt.ylabel('IOU', fontsize=16)
        # plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    plt.xlabel('conv layer number', fontsize=20)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()


    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_err_dist_data = []
        for conv_i in range(num_layers):
            # if conv_i == 0: continue
            df = err_dist_dict[model_name]
            dist_mean = df.loc[df.LAYER == f'conv{conv_i+1}', 'HOT_SPOT_ERR_DIST'].to_numpy().mean()
            avg_err_dist_data.append(dist_mean)

        plt.plot(avg_err_dist_data, '.-', markersize=20)
        # plt.ylim([0, 10])
        plt.ylabel('Avg hot spot error distance (pix)', fontsize=16)
        # plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    plt.xlabel('conv_i', fontsize=20)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()


    plt.figure(figsize=(12, 6))
    for model_name, num_layers in num_layers_dict.items():
        avg_err_dist_data = []
        for conv_i in range(num_layers):
            # if conv_i == 0: continue
            df = err_dist_dict[model_name]
            dist_mean = df.loc[df.LAYER == f'conv{conv_i+1}', 'GAUSSIAN_ERR_DIST'].to_numpy().mean()
            avg_err_dist_data.append(dist_mean)

        plt.plot(avg_err_dist_data, '.-', markersize=20)
        # plt.ylim([0, 10])
        plt.ylabel('Avg Gaussian error distance (pix)', fontsize=16)
        plt.title(f"{map1_name} vs. {map2_name}", fontsize=18)

    max_num_layers = max(num_layers_dict.values())
    plt.xticks(np.arange(max_num_layers), np.arange(1, max_num_layers+1), fontsize=16)
    plt.xlabel('conv_i', fontsize=20)
    plt.legend(num_layers_dict.keys(), fontsize=16)

    pdf.savefig()
    plt.show()
    plt.close()
