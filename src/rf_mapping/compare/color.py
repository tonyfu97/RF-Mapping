"""
To visualize the differences between rfmp4a (white and black) and rfmp4c7o
(color bars).

Notes on abbrevations in this script:
    a- : Achromatic
    c- : Chromatic/Color
    SP : Stimulus Parameters
    CR : Center Responses

Tony Fu, Aug 11, 2022
"""
import os
import sys
import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.result_txt_format import (Rfmp4aTB1 as TB1,
                                              Rfmp4aTBn as TBn,
                                              Rfmp4aNonOverlap as NO,
                                              Rfmp4aWeighted as W,
                                              Rfmp4aSplist as aSP,
                                              Rfmp4c7oSplist as cSP,
                                              CenterReponses as CR,)

# Please specify the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

# Source directories
rfmp4a_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')
rfmp4a_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')
rfmp4c7o_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'mapping')
rfmp4c7o_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'gaussian_fit')

# Result directories
result_dir = os.path.join(c.REPO_DIR, 'results', 'compare')

###############################################################################

# Load the data of RFMP4a (achromatic bars) into pandas DFs.
a_tb1_path   = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb1.txt")
a_tb20_path  = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb20.txt")
a_tb100_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{model_name}_rfmp4a_tb100.txt")
a_no_path  = os.path.join(rfmp4a_fit_dir, model_name, f"non_overlap.txt")
a_w_t_path = os.path.join(rfmp4a_fit_dir, model_name, f"weighted_top.txt")
a_w_b_path = os.path.join(rfmp4a_fit_dir, model_name, f"weighted_bot.txt")

                                                               # Data source                Abbreviation(s)
                                                               # -----------------------    ---------------
a_tb1_df   = pd.read_csv(a_tb1_path,    sep=" ", header=None)  # Top bar                    tb1
a_tb20_df  = pd.read_csv(a_tb20_path,   sep=" ", header=None)  # Avg of top 20 bars         tb20
a_tb100_df = pd.read_csv(a_tb100_path,  sep=" ", header=None)  # Avg of top 100 bars        tb100
a_no_df    = pd.read_csv(a_no_path,     sep=" ", header=None)  # Non-overlap bar maps       no
a_w_t_df   = pd.read_csv(a_w_t_path,    sep=" ", header=None)  # Weighted top bar maps      w_t
a_w_b_df   = pd.read_csv(a_w_b_path,    sep=" ", header=None)  # Weighted bottom bar maps   w_b

# Load the data of RFMP4c7o (Chromatic bars) into pandas DFs.
c_tb1_path   = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb1.txt")
c_tb20_path  = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb20.txt")
c_tb100_path = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb100.txt")
c_no_path  = os.path.join(rfmp4c7o_fit_dir, model_name, f"non_overlap.txt")
c_w_t_path = os.path.join(rfmp4c7o_fit_dir, model_name, f"weighted_top.txt")
c_w_b_path = os.path.join(rfmp4c7o_fit_dir, model_name, f"weighted_bot.txt")

                                                               # Data source                Abbreviation(s)
                                                               # -----------------------    ---------------
c_tb1_df   = pd.read_csv(c_tb1_path,    sep=" ", header=None)  # Top bar                    tb1
c_tb20_df  = pd.read_csv(c_tb20_path,   sep=" ", header=None)  # Avg of top 20 bars         tb20
c_tb100_df = pd.read_csv(c_tb100_path,  sep=" ", header=None)  # Avg of top 100 bars        tb100
c_no_df    = pd.read_csv(c_no_path,     sep=" ", header=None)  # Non-overlap bar maps       no
c_w_t_df   = pd.read_csv(c_w_t_path,    sep=" ", header=None)  # Weighted top bar maps      w_t
c_w_b_df   = pd.read_csv(c_w_b_path,    sep=" ", header=None)  # Weighted bottom bar maps   w_b

# Name the columns.
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]

set_column_names(a_tb1_df, TB1)
set_column_names(a_tb20_df, TBn)
set_column_names(a_tb100_df, TBn)
set_column_names(a_no_df, NO)
set_column_names(a_w_t_df, W)
set_column_names(a_w_b_df, W)

set_column_names(c_tb1_df, TB1)
set_column_names(c_tb20_df, TBn)
set_column_names(c_tb100_df, TBn)
set_column_names(c_no_df, NO)
set_column_names(c_w_t_df, W)
set_column_names(c_w_b_df, W)

# Get/set some model-specific information.
layer_indices, rf_sizes = get_rf_sizes(model, (227, 227))
unit_counter = ConvUnitCounter(model)
_, nums_units = unit_counter.count()
num_layers = len(rf_sizes)
fxvar_thres = 0.8

#######################################.#######################################
#                                                                             #
#            PDF NO.1 DISTRIBUTIONO OF BAR_LENGTH IN TOP/BOTTOM MAPS          #
#                                                                             #
###############################################################################
for conv_i in range(layer_indices):
    layer_name = f"conv{conv_i+1}"
    num_units = nums_units[conv_i]

    # Load layer-specific RFMP4a (Achromatic) data that couldn't be loaded earlier.
    a_splist_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{layer_name}_splist.txt")
    a_t5000_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{layer_name}_top5000_responses.txt")
    a_b5000_path = os.path.join(rfmp4a_mapping_dir, model_name, f"{layer_name}_bot5000_responses.txt")
    
    if os.path.exists(a_splist_path):
        a_splist_df = pd.read_csv(a_splist_path, sep=" ", header=None)
    else:
        break  # In case this layer was not mapped.
    if os.path.exist(a_t5000_path):
        a_t5000_df = pd.read_csv(a_t5000_path, sep=" ", header=None)
    if os.path.exist(a_b5000_path):
        a_b5000_df = pd.read_csv(a_b5000_path, sep=" ", header=None)
    
    # Give the dataframes meaningful headers.
    set_column_names(a_splist_df, aSP)
    set_column_names(a_t5000_df, CR)
    set_column_names(a_b5000_df, CR)
    
    
    # Load layer-specific RFMP4c7o (Chromatic) data that couldn't be loaded earlier.
    c_splist_path = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{layer_name}_splist.txt")
    c_t5000_path = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{layer_name}_top5000_responses.txt")
    c_b5000_path = os.path.join(rfmp4c7o_mapping_dir, model_name, f"{layer_name}_bot5000_responses.txt")

    if os.path.exists(c_splist_path):
        c_splist_df = pd.read_csv(c_splist_path, sep=" ", header=None)
    else:
        break  # In case this layer was not mapped.
    if os.path.exist(c_t5000_path):
        c_t5000_df = pd.read_csv(c_t5000_path, sep=" ", header=None)
    if os.path.exist(c_b5000_path):
        c_b5000_df = pd.read_csv(c_b5000_path, sep=" ", header=None)
    
    # Give the dataframes meaningful headers.
    set_column_names(c_splist_df, cSP)
    set_column_names(c_t5000_df, CR)
    set_column_names(c_b5000_df, CR)

    pdf_path = os.path.join(result_dir, f"{model_name}_{layer_name}_blen_color.pdf")
    with PdfPages(pdf_path) as pdf:
        for unit_i in range(num_units):
            plt.figure(figsize=(12,24))
            plt.suptitle(f"Distribution of Bar Lengths of Top and Bottom 5000 Bars ({model_name} {layer_name} no.{unit_i})", fontsize=18)

            plt.subplot(4,2,1)
            plt.grid()
            blens = sorted(a_splist_df.LEN.unique())
            response_avg = []
            yerror = []
            for blen in blens:
                stim_i = a_t5000_df.loc[(a_t5000_df.UNIT == unit_i) & (a_t5000_df.LEN == blen), 'STIM_I']
                responses_of_this_blen = a_t5000_df[stim_i, 'R']
                response_avg.append(np.mean(responses_of_this_blen))
                yerror.append(np.std(responses_of_this_blen)/math.sqrt(len(responses_of_this_blen)))
            plt.errorbar(blens, response_avg, yerr=yerror, ecolor='black')
            plt.xlabel('bar length (pix)', fontsize=18)
            plt.ylabel('avg response', fontsize=18)
            plt.title(f"RFMP4a top 5000", fontsize=18)

            plt.subplot(4,2,2)
            plt.grid()
            all_blens = a_splist_df.loc[(a_splist_df.UNIT == unit_i), 'LEN']
            plt.hist(all_blens)
            plt.xlabel('bar length (pix)', fontsize=18)
            plt.ylabel('counts', fontsize=18)
            plt.title(f"RFMP4a top 5000", fontsize=18)

            pdf.savefig()
            plt.close()
        



    
    
    