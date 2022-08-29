"""
To visualize the difference between ground truth and bar mapping methods.

Tony Fu, July 27th, 2022
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
from src.rf_mapping.spatial import get_rf_sizes
from src.rf_mapping.result_txt_format import (GtGaussian as GT,
                                              Rfmp4aTB1 as TB1,
                                              Rfmp4aTBn as TBn,
                                              Rfmp4aNonOverlap as NO,
                                              Rfmp4aWeighted as W)

# Please specify the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

# Please specify what ground_truth method versus what RFMP4
is_occlude = False
is_rfmp4a = True

# Source directories
if is_occlude:
    gt_dir = os.path.join(c.REPO_DIR, 'results', 'occlude', 'gaussian_fit')
else:
    gt_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'gaussian_fit')

if is_rfmp4a:
    rfmp4_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')
    rfmp4_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')
else:
    rfmp4_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'mapping')
    rfmp4_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4c7o', 'gaussian_fit')

# Result directories
if is_rfmp4a and not is_occlude:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'gt_vs_rfmp4a', model_name)
elif not is_rfmp4a and not is_occlude:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'gt_vs_rfmp4c7o', model_name)
elif is_rfmp4a and is_occlude:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'occlude_vs_rfmp4a', model_name)
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'occlude_vs_rfmp4c7o', model_name)

###############################################################################

# Load the source files into pandas DFs.
if is_occlude:
    gt_top_path = os.path.join(gt_dir, model_name, f"{model_name}_occlude_gaussian_top.txt")
    gt_bot_path = os.path.join(gt_dir, model_name, f"{model_name}_occlude_gaussian_bot.txt")
else:
    gt_top_path = os.path.join(gt_dir, model_name, 'abs', f"{model_name}_gt_gaussian_top.txt")
    gt_bot_path = os.path.join(gt_dir, model_name, 'abs', f"{model_name}_gt_gaussian_bot.txt")

no_path  = os.path.join(rfmp4_fit_dir, model_name, f"non_overlap.txt")
w_t_path = os.path.join(rfmp4_fit_dir, model_name, f"weighted_top.txt")
w_b_path = os.path.join(rfmp4_fit_dir, model_name, f"weighted_bot.txt")

if is_rfmp4a:
    tb1_path   = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4a_tb1.txt")
    tb20_path  = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4a_tb20.txt")
    tb100_path = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4a_tb100.txt")
else:
    tb1_path   = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb1.txt")
    tb20_path  = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb20.txt")
    tb100_path = os.path.join(rfmp4_mapping_dir, model_name, f"{model_name}_rfmp4c7o_tb100.txt")

                                                           # Data source                Abbreviation(s)
                                                           # -----------------------    ---------------
gt_t_df  = pd.read_csv(gt_top_path, sep=" ", header=None)  # Ground truth top           gt or gt_t
gt_b_df  = pd.read_csv(gt_bot_path, sep=" ", header=None)  # Ground truth bottom        gb or gt_b
tb1_df   = pd.read_csv(tb1_path,    sep=" ", header=None)  # Top bar                    tb1
tb20_df  = pd.read_csv(tb20_path,   sep=" ", header=None)  # Avg of top 20 bars         tb20
tb100_df = pd.read_csv(tb100_path,  sep=" ", header=None)  # Avg of top 100 bars        tb100
no_df    = pd.read_csv(no_path,     sep=" ", header=None)  # Non-overlap bar maps       no
w_t_df   = pd.read_csv(w_t_path,    sep=" ", header=None)  # Weighted top bar maps      w_t
w_b_df   = pd.read_csv(w_b_path,    sep=" ", header=None)  # Weighted bottom bar maps   w_b

# Name the columns.
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]

set_column_names(gt_t_df, GT)
set_column_names(gt_b_df, GT)
set_column_names(tb1_df, TB1)
set_column_names(tb20_df, TBn)
set_column_names(tb100_df, TBn)
set_column_names(no_df, NO)
set_column_names(w_t_df, W)
set_column_names(w_b_df, W)

# Pad the missing layers with NAN because not all layers are mapped.
gt_no_data = gt_t_df[['LAYER', 'UNIT']].copy()  # template df used for padding
def pad_missing_layers(df):
    return pd.merge(gt_no_data, df, how='left')

tb1_df   = pad_missing_layers(tb1_df)
tb20_df  = pad_missing_layers(tb20_df)
tb100_df = pad_missing_layers(tb100_df)
no_df  = pad_missing_layers(no_df)
w_t_df = pad_missing_layers(w_t_df)
w_b_df = pad_missing_layers(w_b_df)


# Get/set some model-specific information.
layer_indices, rf_sizes = get_rf_sizes(model, (227, 227))
num_layers = len(rf_sizes)
fxvar_thres = 0.8

# Name the methods (used for naming pdf files and plot titles & labels)
if is_occlude:
    gt_method = "occlude"
else:
    gt_method = "gt"

if is_rfmp4a:
    ephys_method = "rfmp4a"
else:
    ephys_method = "rfmp4c7o"

#######################################.#######################################
#                                                                             #
#                              PDF NO.0 FXVAR                                 #
#                                                                             #
###############################################################################
def make_fxvar_pdf():
    gt_fxvar = []
    gb_fxvar = []
    w_t_fxvar = []
    w_b_fxvar = []

    gt_labels = []
    gb_labels = []
    w_t_labels = []
    w_b_labels = []

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
        
        w_t_data = w_t_df.loc[(w_t_df.LAYER == layer_name), 'FXVAR']
        w_t_data = w_t_data[np.isfinite(w_t_data)]
        w_t_num_units = len(w_t_data)
        w_t_mean = w_t_data.mean()
        w_t_fxvar.append(w_t_data)
        w_t_labels.append(f"{layer_name}\n(n={w_t_num_units},mu={w_t_mean:.2f})")
        
        w_b_data = w_b_df.loc[(w_b_df.LAYER == layer_name), 'FXVAR']
        w_b_data = w_b_data[np.isfinite(w_b_data)]
        w_b_num_units = len(w_b_data)
        w_b_mean = w_b_data.mean()
        w_b_fxvar.append(w_b_data)
        w_b_labels.append(f"{layer_name}\n(n={w_b_num_units},mu={w_b_mean:.2f})")

    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_and_{ephys_method}_fxvar.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(num_layers*2,20))
        plt.suptitle(f"Fractions of explained variance of {gt_method} and {ephys_method} elliptical Gaussians ({model_name})", fontsize=14)

        plt.subplot(2,1,1)
        plt.grid()
        plt.boxplot(gt_fxvar, labels=gt_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} {gt_method} top", fontsize=18)

        plt.subplot(2,1,2)
        plt.grid()
        plt.boxplot(gb_fxvar, labels=gb_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} {gt_method} bottom", fontsize=18)

        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(num_layers*2,20))
        plt.suptitle(f"Fractions of explained variance of weighted bar maps of {model_name}", fontsize=18)

        plt.subplot(2,1,1)
        plt.grid()
        plt.boxplot(w_t_fxvar, labels=w_t_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} weighted {ephys_method} top", fontsize=18)
        plt.ylim([0, 1])

        plt.subplot(2,1,2)
        plt.grid()
        plt.boxplot(w_b_fxvar, labels=w_b_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} weighted {ephys_method} bottom", fontsize=18)

        pdf.savefig()
        plt.close()

if __name__ == '__main__':
    make_fxvar_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                       PDF NO.1 COORDINATES OF RF CENTERS                    #
#                                                                             #
###############################################################################
def config_plot(limits):
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')

def make_coords_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_and_{ephys_method}_coords.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            rf_size = rf_size[0]
            # limits=(-rf_size//2, rf_size//2)
            limits = (-100, 100)

            plt.figure(figsize=(30,10))
            plt.suptitle(f"{model_name} {layer_name} RF center coordinates of {gt_method} and {ephys_method} (n = {num_units_total})", fontsize=24)

            xdata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,1)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.ylabel('y')
            plt.title(f'{gt_method} (top, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,7)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{gt_method} (bottom, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'TOP_X']
            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'TOP_Y']
            num_units_included = len(xdata)
            xnoise = np.random.rand(num_units_included)
            ynoise = np.random.rand(num_units_included)
            plt.subplot(2,6,2)
            config_plot(limits)
            plt.scatter(xdata + xnoise, ydata + ynoise, alpha=0.4)
            plt.title(f'{ephys_method} (top, n = {num_units_included}, with noise)')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'BOT_X']
            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'BOT_Y']
            num_units_included = len(xdata)
            xnoise = np.random.rand(num_units_included)
            ynoise = np.random.rand(num_units_included)
            plt.subplot(2,6,8)
            config_plot(limits)
            plt.scatter(xdata + xnoise, ydata + ynoise, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'{ephys_method} (bottom, n = {num_units_included}, with noise)')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'TOP_MUX']
            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,3)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'{ephys_method} avg of top 20 bars (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'BOT_MUX']
            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,9)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'{ephys_method} avg of bottom 20 bars (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'TOP_MUX']
            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,4)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'{ephys_method} avg of top 100 bars (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'BOT_MUX']
            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,10)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'{ephys_method} avg of bottom 100 bars (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_X']
            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_Y']
            num_units_included = len(xdata)
            plt.subplot(2,6,5)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'non overlap {ephys_method} (top, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'BOT_X']
            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'BOT_Y']
            num_units_included = len(xdata)
            plt.subplot(2,6,11)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'non overlap {ephys_method} (bottom, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,6)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'weighted {ephys_method} (top, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,12)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'weighted {ephys_method} (bottom, n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()
        
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_coords_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                              PDF NO.2 RF RADIUS                             #
#                                                                             #
###############################################################################
def geo_mean(sd1, sd2):
    return np.sqrt(np.power(sd1, 2) + np.power(sd2, 2))

def make_radius_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_{ephys_method}_radius.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            rf_size = rf_size[0]
            xlim = (0, rf_size)
            ylim = None
            bins = np.linspace(*xlim, 30)

            plt.figure(figsize=(15,10))
            plt.suptitle(f"{model_name} {layer_name} RF radii of {gt_method} and {ephys_method} (n = {num_units_total}, ERF = {rf_size})", fontsize=20)

            sd1data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,1)
            plt.hist(radii, bins=bins)
            plt.ylabel('counts')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'{gt_method} top (n = {num_units_included})')

            sd1data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,4)
            plt.hist(radii, bins=bins)
            plt.xlabel('$\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel('counts')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'{gt_method} bottom (n = {num_units_included})')

            # Note: For poorly fit units, all TOP_RAD is -1, so checking only one of them is sufficient.
            radii_10 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_RAD_10']
            radii_50 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_RAD_50']
            radii_90 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_RAD_90']
            num_units_included = len(radii_10)
            plt.subplot(2,3,2)
            plt.hist(radii_10, bins=bins, label='10% of mass', color=(0.9, 0.5, 0.3, 0.7))
            plt.hist(radii_50, bins=bins, label='50% of mass', color=(0.5, 0.9, 0.5, 0.7))
            plt.hist(radii_90, bins=bins, label='90% of mass', color=(0.3, 0.5, 0.9, 0.7))
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.title(f'non overlap {ephys_method} (top, n = {num_units_included})')

            # Note: For poorly fit units, all BOT_RAD is -1, so checking only one of them is sufficient.
            radii_10 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1), 'BOT_RAD_10']
            radii_50 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1), 'BOT_RAD_50']
            radii_90 = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1), 'BOT_RAD_90']
            num_units_included = len(radii_10)
            plt.subplot(2,3,5)
            plt.hist(radii_10, bins=bins, label='10% of mass', color=(0.9, 0.5, 0.3, 0.7))
            plt.hist(radii_50, bins=bins, label='50% of mass', color=(0.5, 0.9, 0.5, 0.7))
            plt.hist(radii_90, bins=bins, label='90% of mass', color=(0.3, 0.5, 0.9, 0.7))
            plt.xlabel('radius')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
            plt.title(f'non overlap {ephys_method} (bottom, n = {num_units_included})')

            sd1data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,3)
            plt.hist(radii, bins=bins)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'weighted {ephys_method} (top, n = {num_units_included})')

            sd1data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,6)
            plt.hist(radii, bins=bins)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('$\sqrt{sd_1^2+sd_2^2}$')
            plt.title(f'weighted {ephys_method} (bottom, n = {num_units_included})')
            
            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    make_radius_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                          PDF NO.3 RF ORIENTATION                            #
#                                                                             #
###############################################################################
def eccentricity(sd1, sd2):
    short = np.minimum(sd1, sd2)
    long  = np.maximum(sd1, sd2)
    # ecc = np.sqrt(1 - np.power(short, 2)/np.power(long, 2))
    ecc = long/short
    return ecc

def make_ori_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_{ephys_method}_ori.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            ylim = (0, 3)

            plt.figure(figsize=(10,11))
            plt.suptitle(f"{model_name} {layer_name} RF orientation of {gt_method} and {ephys_method}\n(n = {num_units_total})", fontsize=16)

            layer_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(221, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'{gt_method} top (n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(223, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'{gt_method} bottom (n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(222, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'weighted {ephys_method} (top, n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(224, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'weighted {ephys_method} (bottom, n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_ori_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                             PDF NO.4 ERROR COORDS                           #
#                                                                             #
###############################################################################
def config_plot(limits):
    line = np.linspace(min(limits), max(limits), 100)
    plt.plot(line, line, 'k', alpha=0.4)
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')

def make_error_coords_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_vs_{ephys_method}_coords.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            limits = (-100, 100)

            gt_xdata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUX']
            gt_ydata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUY']
            gb_xdata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUX']
            gb_ydata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUY']

            plt.figure(figsize=(25,20))
            plt.suptitle(f"Comparing {model_name} {layer_name} RF center coordinates of {gt_method} and {ephys_method} (n = {num_units_total})", fontsize=24)

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_X']
            num_units_included = len(xdata)
            
            if sum(np.isfinite(xdata)) == 0: 
                continue  # Skip this layer if no data
            
            plt.subplot(4,5,1)
            config_plot(limits)
            plt.scatter(gt_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gt_xdata, xdata)
            plt.title(f'{gt_method} vs. top 1 x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_Y']
            num_units_included = len(xdata)
            plt.subplot(4,5,6)
            config_plot(limits)
            plt.scatter(gt_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gt_ydata, ydata)
            plt.title(f'{gt_method} vs. top 1 y (n={num_units_included}, r={r_val:.2f})')

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,2)
            config_plot(limits)
            plt.scatter(gt_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gt_xdata, xdata)
            plt.title(f'{gt_method} vs. top 20 avg x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,7)
            config_plot(limits)
            plt.scatter(gt_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gt_ydata, ydata)
            plt.title(f'{gt_method} vs. top 20 avg y (n={num_units_included}, r={r_val:.2f})')

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,3)
            config_plot(limits)
            plt.scatter(gt_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gt_xdata, xdata)
            plt.title(f'{gt_method} vs. top 100 avg x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,8)
            config_plot(limits)
            plt.scatter(gt_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gt_ydata, ydata)
            plt.title(f'{gt_method} vs. top 100 avg y (n={num_units_included}, r={r_val:.2f})')

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_X']
            num_units_included = len(xdata)
            plt.subplot(4,5,4)
            config_plot(limits)
            plt.scatter(gt_xdata.loc[no_df.TOP_RAD_10 != -1], xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            try:
                r_val, p_val = pearsonr(gt_xdata.loc[no_df.TOP_RAD_10 != -1], xdata)
            except:
                r_val = -1
            plt.title(f'{gt_method} vs. top non-overlap x (n={num_units_included}, r={r_val:.2f})')

            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_Y']
            num_units_included = len(xdata)
            plt.subplot(4,5,9)
            config_plot(limits)
            plt.scatter(gt_ydata.loc[no_df.TOP_RAD_10 != -1], ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            try:
                r_val, p_val = pearsonr(gt_ydata.loc[no_df.TOP_RAD_10 != -1], ydata)
            except:
                r_val = -1
            plt.title(f'{gt_method} vs. top non-overlap y (n={num_units_included}, r={r_val:.2f})')

            xdata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres) & (gt_t_df.FXVAR > fxvar_thres), 'MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,5)
            config_plot(limits)
            plt.scatter(gt_xdata.loc[w_t_df.FXVAR > fxvar_thres], xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            try:
                r_val, p_val = pearsonr(gt_xdata.loc[w_t_df.FXVAR > fxvar_thres], xdata)
            except:
                r_val = -1
            plt.title(f'{gt_method} vs. top weighted x (n={num_units_included}, r={r_val:.2f})')

            ydata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres) & (gt_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,10)
            config_plot(limits)
            plt.scatter(gt_ydata.loc[w_t_df.FXVAR > fxvar_thres], ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            try:
                r_val, p_val = pearsonr(gt_ydata.loc[w_t_df.FXVAR > fxvar_thres], ydata)
            except:
                r_val = -1
            plt.title(f'{gt_method} vs. top weighted y (n={num_units_included}, r={r_val:.2f})')

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_X']
            num_units_included = len(xdata)
            plt.subplot(4,5,11)
            config_plot(limits)
            plt.scatter(gb_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gb_xdata, xdata)
            plt.title(f'{gt_method} vs. bottom 1 x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_Y']
            num_units_included = len(xdata)
            plt.subplot(4,5,16)
            config_plot(limits)
            plt.scatter(gb_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gb_ydata, ydata)
            plt.title(f'GT vs. bottom 1 y (n={num_units_included}, r={r_val:.2f})')

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,12)
            config_plot(limits)
            plt.scatter(gb_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gb_xdata, xdata)
            plt.title(f'{gt_method} vs. bottom 20 x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,17)
            config_plot(limits)
            plt.scatter(gb_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gb_ydata, ydata)
            plt.title(f'{gt_method} vs. bottom 20 y (n={num_units_included}, r={r_val:.2f})')

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,13)
            config_plot(limits)
            plt.scatter(gb_xdata, xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            r_val, p_val = pearsonr(gb_xdata, xdata)
            plt.title(f'{gt_method} vs. bottom 100 x (n={num_units_included}, r={r_val:.2f})')

            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,18)
            config_plot(limits)
            plt.scatter(gb_ydata, ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            r_val, p_val = pearsonr(gb_ydata, ydata)
            plt.title(f'{gt_method} vs. bottom 100 y (n={num_units_included}, r={r_val:.2f})')

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_X']
            num_units_included = len(xdata)
            plt.subplot(4,5,14)
            config_plot(limits)
            plt.scatter(gb_xdata.loc[no_df.BOT_RAD_10 != -1], xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            try:
                r_val, p_val = pearsonr(gb_xdata.loc[no_df.BOT_RAD_10 != -1], xdata)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. bottom non-overlap x (n={num_units_included}, r={r_val:.2f})')

            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_Y']
            num_units_included = len(xdata)
            plt.subplot(4,5,19)
            config_plot(limits)
            plt.scatter(gb_ydata.loc[no_df.BOT_RAD_10 != -1], ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            try:
                r_val, p_val = pearsonr(gb_ydata.loc[no_df.BOT_RAD_10 != -1], ydata)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. bottom non-overlap y (n={num_units_included}, r={r_val:.2f})')

            xdata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres) & (gt_b_df.FXVAR > fxvar_thres), 'MUX']
            num_units_included = len(xdata)
            plt.subplot(4,5,15)
            config_plot(limits)
            plt.scatter(gb_xdata.loc[w_b_df.FXVAR > fxvar_thres], xdata, alpha=0.4)
            plt.xlabel(f'{gt_method} x')
            plt.ylabel(f'{ephys_method} x')
            try:
                r_val, p_val = pearsonr(gb_xdata.loc[w_b_df.FXVAR > fxvar_thres], xdata)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. bottom weighted x (n={num_units_included}, r={r_val:.2f})')

            ydata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres) & (gt_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(4,5,20)
            config_plot(limits)
            plt.scatter(gb_ydata.loc[w_b_df.FXVAR > fxvar_thres], ydata, alpha=0.4)
            plt.xlabel(f'{gt_method} y')
            plt.ylabel(f'{ephys_method} y')
            try:
                r_val, p_val = pearsonr(gb_ydata.loc[w_b_df.FXVAR > fxvar_thres], ydata)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} GT vs. bottom weighted y (n = {num_units_included}, r = {r_val:.2f})')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_error_coords_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                             PDF NO.5 ERROR RADIUS                           #
#                                                                             #
###############################################################################
def config_plot(limits):
    line = np.linspace(min(limits), max(limits), 100)
    plt.plot(line, line, 'k', alpha=0.4)
    plt.xlim(limits)
    plt.ylim(limits)
    ax = plt.gca()
    ax.set_aspect('equal')

def del_outliers(radius_1, radius_2, rf_size):
    new_radius_1 = []
    new_radius_2 = []
    for i in range(len(radius_1)):
        if radius_1.iloc[i] < rf_size and radius_2.iloc[i] < rf_size:
            new_radius_1.append(radius_1.iloc[i])
            new_radius_2.append(radius_2.iloc[i])
    return np.array(new_radius_1), np.array(new_radius_2)

def make_error_radius_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_vs_{ephys_method}_radius.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            limits = (0, 120)
            rf_size = rf_size[0]

            gt_sd1 = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD1']
            gt_sd2 = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD2']
            gt_radius = geo_mean(gt_sd1, gt_sd2)
            
            gb_sd1 = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD1']
            gb_sd2 = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD2']
            gb_radius = geo_mean(gb_sd1, gb_sd2)

            plt.figure(figsize=(20,10))
            plt.suptitle(f"Comparing {model_name} {layer_name} RF radii of {gt_method} and {ephys_method} (n = {num_units_total}, ERF = {rf_size})", fontsize=18)

            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_RAD_10']
            if sum(np.isfinite(radius)) == 0: 
                continue  # Skip this layer if no data
            plt.subplot(2,4,1)
            config_plot(limits)
            plt.scatter(gt_radius.loc[no_df.TOP_RAD_10 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(f'{ephys_method} 10% of mass')
            try:
                r_val, p_val = pearsonr(gt_radius.loc[no_df.TOP_RAD_10 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. non-overlap (top, n={len(radius)}, r={r_val:.2f})')

            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_50 != -1) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_RAD_50']
            plt.subplot(2,4,2)
            config_plot(limits)
            plt.scatter(gt_radius.loc[no_df.TOP_RAD_50 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(f'{ephys_method} 50% of mass')
            try:
                r_val, p_val = pearsonr(gt_radius.loc[no_df.TOP_RAD_50 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. non-overlap (top, n={len(radius)}, r={r_val:.2f})')

            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_90 != -1) & (gt_t_df.FXVAR > fxvar_thres), 'TOP_RAD_90']
            plt.subplot(2,4,3)
            config_plot(limits)
            plt.scatter(gt_radius.loc[no_df.TOP_RAD_90 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(f'{ephys_method} 90% of mass')
            try:
                r_val, p_val = pearsonr(gt_radius.loc[no_df.TOP_RAD_90 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. non-overlap (top, n={len(radius)}, r={r_val:.2f})')

            sd1 = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres) & (gt_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2 = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres) & (gt_t_df.FXVAR > fxvar_thres), 'SD2']
            radius = geo_mean(sd1, sd2)
            tmp_gt_radius = gt_radius.loc[w_t_df.FXVAR > fxvar_thres]
            tmp_gt_radius, radius = del_outliers(tmp_gt_radius, radius, rf_size)
            plt.subplot(2,4,4)
            config_plot(limits)
            plt.scatter(tmp_gt_radius, radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(ephys_method + ' weighted $\sqrt{sd_1^2+sd_2^2}$')
            try:
                r_val, p_val = pearsonr(tmp_gt_radius, radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. weighted (top, n={len(radius)}, r={r_val:.2f})')
            
            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_10 != -1) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_RAD_10']
            plt.subplot(2,4,5)
            config_plot(limits)
            plt.scatter(gb_radius.loc[no_df.BOT_RAD_10 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(ephys_method + ' 10% of mass')
            try:
                r_val, p_val = pearsonr(gb_radius.loc[no_df.BOT_RAD_10 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. non-overlap (bottom, n={len(radius)}, r={r_val:.2f})')

            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_50 != -1) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_RAD_50']
            plt.subplot(2,4,6)
            config_plot(limits)
            plt.scatter(gb_radius.loc[no_df.BOT_RAD_50 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(ephys_method + ' 50% of mass')
            try:
                r_val, p_val = pearsonr(gb_radius.loc[no_df.BOT_RAD_50 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'GT vs. non-overlap (bottom, n={len(radius)}, r={r_val:.2f})')
            
            radius = no_df.loc[(no_df.LAYER == layer_name) & (no_df.BOT_RAD_90 != -1) & (gt_b_df.FXVAR > fxvar_thres), 'BOT_RAD_90']
            plt.subplot(2,4,7)
            config_plot(limits)
            plt.scatter(gb_radius.loc[no_df.BOT_RAD_90 != -1], radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(ephys_method + ' 90% of mass')
            try:
                r_val, p_val = pearsonr(gb_radius.loc[no_df.BOT_RAD_90 != -1], radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. non-overlap (bottom, n={len(radius)}, r={r_val:.2f})')
            
            sd1 = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres) & (gt_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2 = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres) & (gt_b_df.FXVAR > fxvar_thres), 'SD2']
            radius = geo_mean(sd1, sd2)
            tmp_gb_radius = gb_radius.loc[w_b_df.FXVAR > fxvar_thres]
            tmp_gb_radius, radius = del_outliers(tmp_gb_radius, radius, rf_size)
            plt.subplot(2,4,8)
            config_plot(limits)
            plt.scatter(tmp_gb_radius, radius, alpha=0.4)
            plt.xlabel(gt_method + ' $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel(ephys_method + ' weighted $\sqrt{sd_1^2+sd_2^2}$')
            try:
                r_val, p_val = pearsonr(tmp_gb_radius, radius)
            except:
                r_val = np.NaN
            plt.title(f'{gt_method} vs. weighted (bottom, n={len(radius)}, r={r_val:.2f})')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_error_radius_pdf()
    pass


######################################.#######################################
#                                                                            #
#                         PDF NO.6 ERROR ORIENTATION                         #
#                                                                            #
##############################################################################
def config_plot():
    plt.xlim([-5, 95])
    # plt.ylim([0, 7])
    plt.xlabel('$\Delta \Theta $ (°)')

def delta_ori(ori_1, ori_2):
    # Note: this function assumes 0 <= ori < 180.
    theta_small = np.minimum(ori_1, ori_2)
    theta_large = np.maximum(ori_1, ori_2)
    # Because angles wraps around 0 and 180 deg, we need to consider two cases:
    delta_theta_a = theta_large - theta_small
    delta_theta_b = (theta_small + 180) - theta_large
    return np.minimum(delta_theta_a, delta_theta_b)

annotate_eccentricity_threshold = 3
def annotate_eccentricity(units, angle_diff, eccentricities):
    # Display unid indices for those that have large eccentricity values.
    ax = plt.gca()
    for unit_i, angle, ecc in zip(units, angle_diff, eccentricities):
        if ecc > annotate_eccentricity_threshold:
            ax.annotate(unit_i, (angle, ecc), fontsize=5)

def make_error_ori_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_{gt_method}_vs_{ephys_method}_ori.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            
            # Get ground truth data (top and bottom)
            gt_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (w_t_df.FXVAR > fxvar_thres)]
            gt_ecc = eccentricity(gt_data['SD1'], gt_data['SD2'])
            gt_ori = gt_data['ORI']
            gb_data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (w_b_df.FXVAR > fxvar_thres)]
            gb_ecc = eccentricity(gb_data['SD1'], gb_data['SD2'])
            gb_ori = gb_data['ORI']
            
            # Get weighted maps data (top and bottom)
            w_t_data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (w_t_df.FXVAR > fxvar_thres)]
            w_t_ecc = eccentricity(w_t_data['SD1'], w_t_data['SD2'])
            w_t_ori = w_t_data['ORI']
            w_b_data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (w_b_df.FXVAR > fxvar_thres)]
            w_b_ecc = eccentricity(w_b_data['SD1'], w_b_data['SD2'])
            w_b_ori = w_b_data['ORI']

            plt.figure(figsize=(10,11))
            plt.suptitle(f"Comparing {model_name} {layer_name} RF orientations of {gt_method} and {ephys_method}\n(n = {num_units_total})", fontsize=14)

            plt.subplot(2,2,1)
            angle_diff = delta_ori(gt_ori, w_t_ori)
            plt.scatter(angle_diff, gt_ecc, alpha=0.4)
            config_plot()
            annotate_eccentricity(gt_data['UNIT'], angle_diff, gt_ecc)
            plt.ylabel(f'{gt_method} eccentricity')
            plt.title(f'{gt_method} vs. weighted {ephys_method} (top, n = {len(gt_ori)})')
            
            plt.subplot(2,2,2)
            angle_diff = delta_ori(gt_ori, w_t_ori)
            plt.scatter(angle_diff, w_t_ecc, alpha=0.4)
            config_plot()
            annotate_eccentricity(gt_data['UNIT'], angle_diff, w_t_ecc)
            plt.ylabel(f'{ephys_method} weighted eccentricity')
            plt.title(f'{gt_method} vs. weighted {ephys_method} (top, n = {len(gt_ori)})')
            
            plt.subplot(2,2,3)
            angle_diff = delta_ori(gb_ori, w_b_ori)
            plt.scatter(angle_diff, gb_ecc, alpha=0.4)
            config_plot()
            annotate_eccentricity(gb_data['UNIT'], angle_diff, gb_ecc)
            plt.ylabel(f'{gt_method} eccentricity')
            plt.title(f'{gt_method} vs. weighted {ephys_method} (bottom, n = {len(gb_ori)})')
            
            plt.subplot(2,2,4)
            angle_diff = delta_ori(gb_ori, w_b_ori)
            plt.scatter(angle_diff, w_b_ecc, alpha=0.4)
            config_plot()
            annotate_eccentricity(gb_data['UNIT'], angle_diff, w_b_ecc)
            plt.ylabel(f'{ephys_method} weighted eccentricity')
            plt.title(f'{gt_method} vs. weighted {ephys_method} (bottom, n = {len(gb_ori)})')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_error_ori_pdf()
    pass
