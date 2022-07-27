import os
import sys
from enum import Enum

import numpy as np
import pandas as pd

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import (GtGaussian as GT,
                                              Rfmp4aNonOverlap as NO,
                                              Rfmp4aTb as W,
                                              Rfmp4aTb as TB)

model_name = 'alexnet'

# Source directories
gt_dir             = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'gaussian_fit')
rfmp4a_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')
rfmp4a_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')

# Result directories
result_dir = os.path.join(c.REPO_DIR, 'results', 'compare')


###############################################################################

gt_top_path = os.path.join(gt_dir, model_name, f"{model_name}_gt_gaussian_top.txt")
gt_bot_path = os.path.join(gt_dir, model_name, f"{model_name}_gt_gaussian_bot.txt")
t1_path   = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb1.txt")
t20_path  = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb20.txt")
t100_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb100.txt")
no_path  = os.path.join(rfmp4a_fit_dir, model_name, f"rfmp4a_non_overlap.txt")
w_t_path = os.path.join(rfmp4a_fit_dir, model_name, f"rfmp4a_weighted_top.txt")
w_b_path = os.path.join(rfmp4a_fit_dir, model_name, f"rfmp4a_weighted_bot.txt")

gt_top_df = pd.read_csv(gt_top_path, sep=" ")
gt_bot_df = pd.read_csv(gt_bot_path, sep=" ")
t1_df   = pd.read_csv(t1_path, sep=" ")
t20_df  = pd.read_csv(t20_path, sep=" ")
t100_df = pd.read_csv(t100_path, sep=" ")
no_df  = pd.read_csv(no_path, sep=" ")
w_t_df = pd.read_csv(w_t_path, sep=" ")
w_b_df = pd.read_csv(w_b_path, sep=" ")



    
    
    


