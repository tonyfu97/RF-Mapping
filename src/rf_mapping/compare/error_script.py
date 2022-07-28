import os
import sys
from enum import Enum

import numpy as np
import pandas as pd

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import (GtGaussian as GT,
                                              Rfmp4aTB1 as TB1,
                                              Rfmp4aTBn as TBn,
                                              Rfmp4aNonOverlap as NO,
                                              Rfmp4aWeighted as W)

model_name = 'alexnet'

# Source directories
gt_dir             = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'gaussian_fit')
rfmp4a_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')
rfmp4a_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')

# Result directories
result_dir = os.path.join(c.REPO_DIR, 'results', 'compare')


###############################################################################

gt_top_path = os.path.join(gt_dir, model_name, 'abs', f"{model_name}_gt_gaussian_top.txt")
gt_bot_path = os.path.join(gt_dir, model_name, 'abs', f"{model_name}_gt_gaussian_bot.txt")
tb1_path   = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb1.txt")
tb20_path  = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb20.txt")
tb100_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb100.txt")
no_path  = os.path.join(rfmp4a_fit_dir, model_name, f"non_overlap.txt")
w_t_path = os.path.join(rfmp4a_fit_dir, model_name, f"weighted_top.txt")
w_b_path = os.path.join(rfmp4a_fit_dir, model_name, f"weighted_bot.txt")

gt_t_df = pd.read_csv(gt_top_path, sep=" ")
gt_b_df = pd.read_csv(gt_bot_path, sep=" ")
tb1_df   = pd.read_csv(tb1_path, sep=" ")
tb20_df  = pd.read_csv(tb20_path, sep=" ")
tb100_df = pd.read_csv(tb100_path, sep=" ")
no_df  = pd.read_csv(no_path, sep=" ")
w_t_df = pd.read_csv(w_t_path, sep=" ")
w_b_df = pd.read_csv(w_b_path, sep=" ")


def set_column_names(df, Format):
    df.columns = [e.name for e in Format]

set_column_names(gt_t_df, GT)
set_column_names(gt_b_df, GT)
set_column_names(tb1_df, TB1)
set_column_names(tb20_df, TBn)
set_column_names(tb100_df, TBn)
set_column_names(no_df, NO)
set_column_names(w_t_df, W)
set_column_names(w_b_df, W)

layer_name = 'conv2'
layer_gt_t_df = gt_t_df[gt_t_df['LAYER'] == layer_name]
layer_gt_b_df = gt_b_df[gt_b_df['LAYER'] == layer_name]
num_units = len(layer_gt_t_df)




