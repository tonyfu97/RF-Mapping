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
top_n = 1000

# Source directories
rfmp4a_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')
rfmp4a_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')
rfmp4c7o_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'mapping')
rfmp4c7o_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'gaussian_fit')

# Result directories
result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'color')

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
#  To see if it is possible to reduce the number of bars by eliminating the   #
#  the small bars.                                                            #
#                                                                             #
###############################################################################
def make_blen_color_pdf():
    for conv_i in range(num_layers):
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
        if os.path.exists(a_t5000_path):
            a_t5000_df = pd.read_csv(a_t5000_path, sep=" ", header=None)
        if os.path.exists(a_b5000_path):
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
        if os.path.exists(c_t5000_path):
            c_t5000_df = pd.read_csv(c_t5000_path, sep=" ", header=None)
        if os.path.exists(c_b5000_path):
            c_b5000_df = pd.read_csv(c_b5000_path, sep=" ", header=None)
        
        # Give the dataframes meaningful headers.
        set_column_names(c_splist_df, cSP)
        set_column_names(c_t5000_df, CR)
        set_column_names(c_b5000_df, CR)

        pdf_path = os.path.join(result_dir, f"{model_name}_{layer_name}_blen_color.pdf")
        with PdfPages(pdf_path) as pdf:
            for unit_i in range(num_units):
                plt.figure(figsize=(24,12))
                plt.suptitle(f"Distribution of Bar Lengths of Top and Bottom {top_n} Bars ({model_name} {layer_name} no.{unit_i})", fontsize=18)

                plt.subplot(2,4,1)
                plt.grid()
                blens = sorted(a_splist_df.LEN.unique())
                xlim = [min(blens)-2, max(blens)+2]
                response_avg = []
                yerror = []
                for blen in blens:
                    stim_i = a_splist_df.loc[a_splist_df.LEN == blen, 'STIM_I']
                    responses_of_this_blen = a_t5000_df.loc[(a_t5000_df.UNIT == unit_i) & (a_t5000_df.STIM_I.isin(stim_i)) & (a_t5000_df.RANK < top_n), 'R']
                    response_avg.append(np.mean(responses_of_this_blen))
                    if len(responses_of_this_blen) == 0:
                        yerror.append(0)
                    else:
                        yerror.append(np.std(responses_of_this_blen)/math.sqrt(len(responses_of_this_blen)))
                plt.errorbar(blens, response_avg, yerr=yerror, ecolor='black')
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('avg response', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4a top", fontsize=18)

                plt.subplot(2,4,2)
                plt.grid()
                stim_i = a_t5000_df.loc[(a_t5000_df.UNIT == unit_i) & (a_t5000_df.RANK < top_n), 'STIM_I']
                all_blens = a_splist_df.loc[a_splist_df.STIM_I.isin(stim_i), 'LEN']
                plt.hist(all_blens)
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('counts', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4a top", fontsize=18)

                plt.subplot(2,4,3)
                plt.grid()
                blens = sorted(c_splist_df.LEN.unique())
                response_avg = []
                yerror = []
                for blen in blens:
                    stim_i = c_splist_df.loc[c_splist_df.LEN == blen, 'STIM_I']
                    responses_of_this_blen = c_t5000_df.loc[(c_t5000_df.UNIT == unit_i) & (c_t5000_df.STIM_I.isin(stim_i)) & (c_t5000_df.RANK < top_n), 'R']
                    response_avg.append(np.mean(responses_of_this_blen))
                    if len(responses_of_this_blen) == 0:
                        yerror.append(0)
                    else:
                        yerror.append(np.std(responses_of_this_blen)/math.sqrt(len(responses_of_this_blen)))
                plt.errorbar(blens, response_avg, yerr=yerror, ecolor='black')
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('avg response', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4c7o top", fontsize=18)

                plt.subplot(2,4,4)
                plt.grid()
                stim_i = c_t5000_df.loc[(c_t5000_df.UNIT == unit_i) & (c_t5000_df.RANK < top_n), 'STIM_I']
                all_blens = c_splist_df.loc[c_splist_df.STIM_I.isin(stim_i), 'LEN']
                plt.hist(all_blens)
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('counts', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4c7o top", fontsize=18)

                plt.subplot(2,4,5)
                plt.grid()
                blens = sorted(a_splist_df.LEN.unique())
                response_avg = []
                yerror = []
                for blen in blens:
                    stim_i = a_splist_df.loc[a_splist_df.LEN == blen, 'STIM_I']
                    responses_of_this_blen = a_b5000_df.loc[(a_b5000_df.UNIT == unit_i) & (a_b5000_df.STIM_I.isin(stim_i)) & (a_b5000_df.RANK < top_n), 'R']
                    response_avg.append(np.mean(responses_of_this_blen))
                    if len(responses_of_this_blen) == 0:
                        yerror.append(0)
                    else:
                        yerror.append(np.std(responses_of_this_blen)/math.sqrt(len(responses_of_this_blen)))
                plt.errorbar(blens, response_avg, yerr=yerror, ecolor='black')
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('avg response', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4a bottom", fontsize=18)

                plt.subplot(2,4,6)
                plt.grid()
                stim_i = a_b5000_df.loc[(a_b5000_df.UNIT == unit_i) & (a_b5000_df.RANK < top_n), 'STIM_I']
                all_blens = a_splist_df.loc[a_splist_df.STIM_I.isin(stim_i), 'LEN']
                plt.hist(all_blens)
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('counts', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4a bottom", fontsize=18)

                plt.subplot(2,4,7)
                plt.grid()
                blens = sorted(c_splist_df.LEN.unique())
                response_avg = []
                yerror = []
                for blen in blens:
                    stim_i = c_splist_df.loc[c_splist_df.LEN == blen, 'STIM_I']
                    responses_of_this_blen = c_b5000_df.loc[(c_b5000_df.UNIT == unit_i) & (c_b5000_df.STIM_I.isin(stim_i)) & (c_b5000_df.RANK < top_n), 'R']
                    response_avg.append(np.mean(responses_of_this_blen))
                    if len(responses_of_this_blen) == 0:
                        yerror.append(0)
                    else:
                        yerror.append(np.std(responses_of_this_blen)/math.sqrt(len(responses_of_this_blen)))
                plt.errorbar(blens, response_avg, yerr=yerror, ecolor='black')
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('avg response', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4c7o bottom", fontsize=18)

                plt.subplot(2,4,8)
                plt.grid()
                stim_i = c_b5000_df.loc[(c_b5000_df.UNIT == unit_i) & (c_b5000_df.RANK < top_n), 'STIM_I']
                all_blens = c_splist_df.loc[c_splist_df.STIM_I.isin(stim_i), 'LEN']
                plt.hist(all_blens)
                plt.xlabel('bar length (pix)', fontsize=12)
                plt.ylabel('counts', fontsize=12)
                plt.xlim(xlim)
                plt.title(f"RFMP4c7o bottom", fontsize=18)

                pdf.savefig()
                plt.close()

if __name__ == "__main__":
    # make_blen_color_pdf()
    pass


#######################################.#######################################
#                                                                             #
#       PDF NO.2 COMPARING MAX and MIN ACTIVATION OF RFMP4a and RFMP4c7o      #
#                                                                             #
#  To see how important colors are.                                           #
#                                                                             #
###############################################################################
annotate_threshold = 20  # Display unit number if (Chromatic_r / Achromatic_r)
                         # is greater than this threshold.

def config_plot(limits):
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    plt.xlabel('RFMP4a')
    plt.ylabel('RFMP4c7o')
    plt.xlim(limits)
    plt.ylim(limits)
    plt.plot(limits, limits, '-', color=(0, 0, 0, 0.5))
    ax = plt.gca()
    ax.set_aspect('equal')

def make_tb1_r_color_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_tb1_r_color.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(5*num_layers,10))
        plt.suptitle(f"Top and bottom 5000 Responses ({model_name})", fontsize=18)

        for conv_i in range(num_layers):
            layer_name = f"conv{conv_i+1}"

            a_top_r = a_tb1_df.loc[a_tb1_df.LAYER == layer_name, 'TOP_R']
            a_bot_r = a_tb1_df.loc[a_tb1_df.LAYER == layer_name, 'BOT_R']
            c_top_r = c_tb1_df.loc[c_tb1_df.LAYER == layer_name, 'TOP_R']
            c_bot_r = c_tb1_df.loc[c_tb1_df.LAYER == layer_name, 'BOT_R']

            plt.subplot(2,num_layers,conv_i+1)
            plt.scatter(a_top_r, c_top_r, alpha=0.4)
            limits = (min(a_top_r.min(), c_top_r.min())-2, max(a_top_r.max(), c_top_r.max())+2)
            config_plot(limits)
            try:
                r_val, p_val = pearsonr(a_top_r, c_top_r)
            except:
                r_val = np.NaN
            plt.title(f"{layer_name} (top, r = {r_val:.2f})", fontsize=16)
            # Display unit idx if c_top_r/a_top_r > annotate_threshold.
            ax = plt.gca()
            for unit_i, (ar, cr) in enumerate(zip(a_top_r, c_top_r)):
                if cr/ar > annotate_threshold:
                    ax.annotate(unit_i, (ar, cr))

            plt.subplot(2,num_layers,conv_i+num_layers+1)
            plt.scatter(a_bot_r, c_bot_r, alpha=0.4)
            limits = (min(a_bot_r.min(), c_bot_r.min())-2, max(a_bot_r.max(), c_bot_r.max())+2)
            config_plot(limits)
            try:
                r_val, p_val = pearsonr(a_bot_r, c_bot_r)
            except:
                r_val = np.NaN
            plt.title(f"{layer_name} (bottom, r = {r_val:.2f})", fontsize=16)
            # Display unit idx if c_top_r/a_top_r > annotate_threshold.
            ax = plt.gca()
            for unit_i, (ar, cr) in enumerate(zip(a_bot_r, c_bot_r)):
                if cr/ar > annotate_threshold:
                    ax.annotate(unit_i, (ar, cr))

        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    # make_tb1_r_color_pdf()
    pass


#######################################.#######################################
#                                                                             #
#             PDF NO.3 COMPARING COORINATES OF RFMP4a and RFMP4c7o            #
#                                                                             #
###############################################################################
def config_plot(limits):
    plt.plot(limits, limits, '-', color=(0, 0, 0, 0.5))
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')

def make_error_coords_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_coords_color.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(a_tb100_df.loc[(a_tb100_df.LAYER == layer_name)])
            limits = (-200, 200)

            plt.figure(figsize=(25,20))
            plt.suptitle(f"RF center coordinates ({model_name} {layer_name}, n = {num_units_total})", fontsize=18)
            
            # if sum(np.isfinite(xddata)) == 0:
            #     continue  # Skip this layer if no data
            
            plt.subplot(4,5,1)
            config_plot(limits)
            a_data = a_tb1_df.loc[(a_tb1_df.LAYER == layer_name), 'TOP_X']
            c_data = c_tb1_df.loc[(c_tb1_df.LAYER == layer_name), 'TOP_X']
            num_units_included = len(a_data)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top x')
            plt.ylabel('RFMP4c7o top x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 1 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb1_df.loc[(a_tb1_df.LAYER == layer_name), 'TOP_Y']
            c_data = c_tb1_df.loc[(c_tb1_df.LAYER == layer_name), 'TOP_Y']
            num_units_included = len(a_data)
            plt.subplot(4,5,6)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top y')
            plt.ylabel('RFMP4c7o top y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 1 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb20_df.loc[(a_tb20_df.LAYER == layer_name), 'TOP_MUX']
            c_data = c_tb20_df.loc[(c_tb20_df.LAYER == layer_name), 'TOP_MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,2)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top mu_x')
            plt.ylabel('RFMP4c7o top mu_x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 20 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb20_df.loc[(a_tb20_df.LAYER == layer_name), 'TOP_MUY']
            c_data = c_tb20_df.loc[(c_tb20_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,7)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top mu_y')
            plt.ylabel('RFMP4c7o top mu_y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 20 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb100_df.loc[(a_tb100_df.LAYER == layer_name), 'TOP_MUX']
            c_data = c_tb100_df.loc[(c_tb100_df.LAYER == layer_name), 'TOP_MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,3)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top mu_x')
            plt.ylabel('RFMP4c7o top mu_x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 100 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb100_df.loc[(a_tb100_df.LAYER == layer_name), 'TOP_MUY']
            c_data = c_tb100_df.loc[(c_tb100_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,8)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top mu_y')
            plt.ylabel('RFMP4c7o top mu_y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top 100 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_no_df.loc[(a_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'TOP_X']
            c_data = c_no_df.loc[(c_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'TOP_X']
            num_units_included = len(a_data)
            plt.subplot(4,5,4)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top x')
            plt.ylabel('RFMP4c7o top x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top non-overlap x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_no_df.loc[(a_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'TOP_Y']
            c_data = c_no_df.loc[(c_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'TOP_Y']
            num_units_included = len(a_data)
            plt.subplot(4,5,9)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top y')
            plt.ylabel('RFMP4c7o top y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top non-overlap y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_w_t_df.loc[(a_w_t_df.LAYER == layer_name) & (a_w_t_df.FXVAR > fxvar_thres) & (c_w_t_df.FXVAR > fxvar_thres), 'MUX']
            c_data = c_w_t_df.loc[(c_w_t_df.LAYER == layer_name) & (a_w_t_df.FXVAR > fxvar_thres) & (c_w_t_df.FXVAR > fxvar_thres), 'MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,5)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top x')
            plt.ylabel('RFMP4c7o top x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top weighted x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_w_t_df.loc[(a_w_t_df.LAYER == layer_name) & (a_w_t_df.FXVAR > fxvar_thres) & (c_w_t_df.FXVAR > fxvar_thres), 'MUY']
            c_data = c_w_t_df.loc[(c_w_t_df.LAYER == layer_name) & (a_w_t_df.FXVAR > fxvar_thres) & (c_w_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,10)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a top y')
            plt.ylabel('RFMP4c7o top y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'top weighted y-coord (n={num_units_included}, r={r_val:.2f})')

            plt.subplot(4,5,11)
            config_plot(limits)
            a_data = a_tb1_df.loc[(a_tb1_df.LAYER == layer_name), 'TOP_Y']
            c_data = c_tb1_df.loc[(c_tb1_df.LAYER == layer_name), 'TOP_Y']
            num_units_included = len(a_data)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bottom x')
            plt.ylabel('RFMP4c7o bottom x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom 1 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb1_df.loc[(a_tb1_df.LAYER == layer_name), 'BOT_Y']
            c_data = c_tb1_df.loc[(c_tb1_df.LAYER == layer_name), 'BOT_Y']
            num_units_included = len(a_data)
            plt.subplot(4,5,16)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bottom y')
            plt.ylabel('RFMP4c7o bottom y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom 1 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb20_df.loc[(a_tb20_df.LAYER == layer_name), 'BOT_MUX']
            c_data = c_tb20_df.loc[(c_tb20_df.LAYER == layer_name), 'BOT_MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,12)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot mu_x')
            plt.ylabel('RFMP4c7o bot mu_x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom 20 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb20_df.loc[(a_tb20_df.LAYER == layer_name), 'BOT_MUY']
            c_data = c_tb20_df.loc[(c_tb20_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,17)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot mu_y')
            plt.ylabel('RFMP4c7o bot mu_y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom 20 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb100_df.loc[(a_tb100_df.LAYER == layer_name), 'BOT_MUX']
            c_data = c_tb100_df.loc[(c_tb100_df.LAYER == layer_name), 'BOT_MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,13)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot mu_x')
            plt.ylabel('RFMP4c7o bot mu_x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bot 100 x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_tb100_df.loc[(a_tb100_df.LAYER == layer_name), 'BOT_MUY']
            c_data = c_tb100_df.loc[(c_tb100_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,18)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot mu_y')
            plt.ylabel('RFMP4c7o bot mu_y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom 100 y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_no_df.loc[(a_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'BOT_X']
            c_data = c_no_df.loc[(c_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'BOT_X']
            num_units_included = len(a_data)
            plt.subplot(4,5,14)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot x')
            plt.ylabel('RFMP4c7o bot x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom non-overlap x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_no_df.loc[(a_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'BOT_Y']
            c_data = c_no_df.loc[(c_no_df.LAYER == layer_name) & (a_no_df.TOP_RAD_10 != -1) &  (c_no_df.TOP_RAD_10 != -1), 'BOT_Y']
            num_units_included = len(a_data)
            plt.subplot(4,5,19)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot y')
            plt.ylabel('RFMP4c7o bot y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom non-overlap y-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_w_b_df.loc[(a_w_b_df.LAYER == layer_name) & (a_w_b_df.FXVAR > fxvar_thres) & (c_w_b_df.FXVAR > fxvar_thres), 'MUX']
            c_data = c_w_b_df.loc[(c_w_b_df.LAYER == layer_name) & (a_w_b_df.FXVAR > fxvar_thres) & (c_w_b_df.FXVAR > fxvar_thres), 'MUX']
            num_units_included = len(a_data)
            plt.subplot(4,5,15)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot x')
            plt.ylabel('RFMP4c7o bot x')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom weighted x-coord (n={num_units_included}, r={r_val:.2f})')

            a_data = a_w_b_df.loc[(a_w_b_df.LAYER == layer_name) & (a_w_b_df.FXVAR > fxvar_thres) & (c_w_b_df.FXVAR > fxvar_thres), 'MUY']
            c_data = c_w_b_df.loc[(c_w_b_df.LAYER == layer_name) & (a_w_b_df.FXVAR > fxvar_thres) & (c_w_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(a_data)
            plt.subplot(4,5,20)
            config_plot(limits)
            plt.scatter(a_data, c_data, alpha=0.4)
            plt.xlabel('RFMP4a bot y')
            plt.ylabel('RFMP4c7o bot y')
            r_val, p_val = pearsonr(a_data, c_data)
            plt.title(f'bottom weighted y-coord (n={num_units_included}, r={r_val:.2f})')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_error_coords_pdf()
    pass



