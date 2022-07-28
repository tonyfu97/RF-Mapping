import os
import sys
import math
from enum import Enum

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

# Source directories
gt_dir             = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'gaussian_fit')
rfmp4a_mapping_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping')
rfmp4a_fit_dir     = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit')

# Result directories
result_dir = os.path.join(c.REPO_DIR, 'results', 'compare')


###############################################################################

# Load the source files into pandas DFs.
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


# Get/set some model-specific information.
layer_indices, rf_sizes = get_rf_sizes(model, (227, 227))
num_layers = len(rf_sizes)
fxvar_thres = 0.8

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
    pdf_path = os.path.join(result_dir, f"{model_name}_coords.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            rf_size = rf_size[0]
            # limits=(-rf_size//2, rf_size//2)
            limits = (-50, 50)

            plt.figure(figsize=(30,10))
            plt.suptitle(f"Estimations of {model_name} {layer_name} RF center coordinates using different techniques", fontsize=24)

            xdata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,1)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.ylabel('y')
            plt.title(f'ground truth top (n = {num_units_included}/{num_units_total})')

            xdata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,7)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'ground truth bot (n = {num_units_included}/{num_units_total})')

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'TOP_X']
            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'TOP_Y']
            num_units_included = len(xdata)
            xnoise = np.random.rand(num_units_included)
            ynoise = np.random.rand(num_units_included)
            plt.subplot(2,6,2)
            config_plot(limits)
            plt.scatter(xdata + xnoise, ydata + ynoise, alpha=0.4)
            plt.title(f'top bar (n = {num_units_included}/{num_units_total}, with noise)')

            xdata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'BOT_X']
            ydata = tb1_df.loc[(tb1_df.LAYER == layer_name), 'BOT_Y']
            num_units_included = len(xdata)
            xnoise = np.random.rand(num_units_included)
            ynoise = np.random.rand(num_units_included)
            plt.subplot(2,6,8)
            config_plot(limits)
            plt.scatter(xdata + xnoise, ydata + ynoise, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'bottom bar (n = {num_units_included}/{num_units_total}, with noise)')

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'TOP_MUX']
            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,3)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'avg of top 20 bars (n = {num_units_included}/{num_units_total})')

            xdata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'BOT_MUX']
            ydata = tb20_df.loc[(tb20_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,9)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'avg of bottom 20 bars (n = {num_units_included}/{num_units_total})')

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'TOP_MUX']
            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'TOP_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,4)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'avg of top 100 bars (n = {num_units_included}/{num_units_total})')

            xdata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'BOT_MUX']
            ydata = tb100_df.loc[(tb100_df.LAYER == layer_name), 'BOT_MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,10)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'avg of bottom 100 bars (n = {num_units_included}/{num_units_total})')

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_X']
            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'TOP_Y']
            num_units_included = len(xdata)
            plt.subplot(2,6,5)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'non overlap bar maps (top, n = {num_units_included}/{num_units_total})')

            xdata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'BOT_X']
            ydata = no_df.loc[(no_df.LAYER == layer_name) & (no_df.TOP_RAD_10 != -1), 'BOT_Y']
            num_units_included = len(xdata)
            plt.subplot(2,6,11)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'non overlap bar maps(bottom, n = {num_units_included}/{num_units_total})')

            xdata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,6)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.title(f'weighted bar maps (top, n = {num_units_included}/{num_units_total})')

            xdata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,6,12)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.title(f'weighted bar maps (bottom, n = {num_units_included}/{num_units_total})')
        
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    # make_coords_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                              PDF NO.2 RF RADIUS                             #
#                                                                             #
###############################################################################

def geo_mean(sd1, sd2):
    return np.sqrt(np.power(sd1, 2) + np.power(sd2, 2))

def make_radius_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_radius.pdf")
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
            plt.suptitle(f"Estimations of {model_name} {layer_name} RF radii using different techniques", fontsize=24)

            sd1data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,1)
            plt.hist(radii, bins=bins)
            plt.ylabel('counts')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'ground truth top (n = {num_units_included}/{num_units_total})')

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
            plt.title(f'ground truth bottom (n = {num_units_included}/{num_units_total})')

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
            plt.title(f'non overlap map (top, n = {num_units_included}/{num_units_total})')

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
            plt.title(f'non overlap map (bottom, n = {num_units_included}/{num_units_total})')

            sd1data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = w_t_df.loc[(w_t_df.LAYER == layer_name) & (w_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,3)
            plt.hist(radii, bins=bins)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'weighted map (top, n = {num_units_included}/{num_units_total})')

            sd1data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = w_b_df.loc[(w_b_df.LAYER == layer_name) & (w_b_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,3,6)
            plt.hist(radii, bins=bins)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('$\sqrt{sd_1^2+sd_2^2}$')
            plt.title(f'weighted map (bottom, n = {num_units_included}/{num_units_total})')
            
            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    # make_radius_pdf()
    pass


#######################################.#######################################
#                                                                             #
#                          PDF NO.3 RF ORIENTATION                            #
#                                                                             #
###############################################################################

def make_ori_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_ori.pdf")
    # with PdfPages(pdf_path) as pdf:
        # for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            # layer_name = f'conv{conv_i+1}'
    layer_name = 'conv2'
    num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])

    plt.figure(figsize=(15,10))
    plt.suptitle(f"Estimations of {model_name} {layer_name} RF orientation using different techniques", fontsize=20)

    layer_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres)]
    num_units_included = len(layer_data)
    eccentricity = np.l
    plt.subplot(2,3,1)
    plt.polar()
    plt.ylabel('counts')
    plt.title(f'ground truth top (n = {num_units_included}/{num_units_total})\n'
              'radius = eccentricity')


if __name__ == '__main__':
    make_ori_pdf()
    pass