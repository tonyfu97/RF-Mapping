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
from src.rf_mapping.result_txt_format import GtGaussian as GT

# Please specify the model
# model = models.alexnet()
# model_name = 'alexnet'
model = models.vgg16()
model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'
this_is_a_test_run = False


# Source directories
gt_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'gaussian_fit', model_name)
oc_dir = os.path.join(c.REPO_DIR, 'results', 'occlude', 'gaussian_fit', model_name)

# Result directories
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'gt_vs_occlude', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'compare', 'gt_vs_occlude', model_name)

###############################################################################

# Load the source files into pandas DFs.
gt_top_path = os.path.join(gt_dir, 'abs', f"{model_name}_gt_gaussian_top.txt")
gt_bot_path = os.path.join(gt_dir, 'abs', f"{model_name}_gt_gaussian_bot.txt")
oc_top_path = os.path.join(oc_dir, f"{model_name}_occlude_gaussian_top.txt")
oc_bot_path = os.path.join(oc_dir, f"{model_name}_occlude_gaussian_bot.txt")

                                                          # Data source                Abbreviation(s)
                                                          # -----------------------    ---------------
gt_t_df = pd.read_csv(gt_top_path, sep=" ", header=None)  # Ground truth top           gt or gt_t
gt_b_df = pd.read_csv(gt_bot_path, sep=" ", header=None)  # Ground truth bottom        gb or gt_b
oc_t_df = pd.read_csv(oc_top_path, sep=" ", header=None)  # Occluder top               ot or oc_t
oc_b_df = pd.read_csv(oc_bot_path, sep=" ", header=None)  # Occluder bottom            ob or oc_b

# Name the columns.
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]

set_column_names(gt_t_df, GT)
set_column_names(gt_b_df, GT)
set_column_names(oc_t_df, GT)
set_column_names(oc_b_df, GT)


# Pad the missing layers with NAN because not all layers are mapped.
gt_no_data = gt_t_df[['LAYER', 'UNIT']].copy()  # template df used for padding
def pad_missing_layers(df):
    return pd.merge(gt_no_data, df, how='left')

oc_t_df = pad_missing_layers(oc_t_df)
oc_b_df = pad_missing_layers(oc_b_df)


# Get/set some model-specific information.
layer_indices, rf_sizes = get_rf_sizes(model, (227, 227))
num_layers = len(rf_sizes)
fxvar_thres = 0.8

#######################################.#######################################
#                                                                             #
#                              PDF NO.0 FXVAR                                 #
#                                                                             #
###############################################################################
def make_fxvar_pdf():
    gt_fxvar = []
    gb_fxvar = []
    oc_t_fxvar = []
    oc_b_fxvar = []

    gt_labels = []
    gb_labels = []
    oc_t_labels = []
    oc_b_labels = []

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
        
        oc_t_data = oc_t_df.loc[(oc_t_df.LAYER == layer_name), 'FXVAR']
        oc_t_data = oc_t_data[np.isfinite(oc_t_data)]
        oc_t_num_units = len(oc_t_data)
        oc_t_mean = oc_t_data.mean()
        oc_t_fxvar.append(oc_t_data)
        oc_t_labels.append(f"{layer_name}\n(n={oc_t_num_units},mu={oc_t_mean:.2f})")
        
        oc_b_data = oc_b_df.loc[(oc_b_df.LAYER == layer_name), 'FXVAR']
        oc_b_data = oc_b_data[np.isfinite(oc_b_data)]
        oc_b_num_units = len(oc_b_data)
        oc_b_mean = oc_b_data.mean()
        oc_b_fxvar.append(oc_b_data)
        oc_b_labels.append(f"{layer_name}\n(n={oc_b_num_units},mu={oc_b_mean:.2f})")

    pdf_path = os.path.join(result_dir, f"{model_name}_fxvar.pdf")
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(num_layers*2,20))
        plt.suptitle(f"Fractions of explained variance of ground truth of {model_name}", fontsize=18)

        plt.subplot(2,1,1)
        plt.grid()
        plt.boxplot(gt_fxvar, labels=gt_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} ground truth top", fontsize=18)

        plt.subplot(2,1,2)
        plt.grid()
        plt.boxplot(gb_fxvar, labels=gb_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} ground truth bottom", fontsize=18)

        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(num_layers*2,20))
        plt.suptitle(f"Fractions of explained variance of occluder discrepancy maps of {model_name}", fontsize=18)

        plt.subplot(2,1,1)
        plt.grid()
        plt.boxplot(oc_t_fxvar, labels=oc_t_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} weighted bar map top", fontsize=18)
        plt.ylim([0, 1])

        plt.subplot(2,1,2)
        plt.grid()
        plt.boxplot(oc_b_fxvar, labels=oc_b_labels, showmeans=True)
        plt.ylabel('fraction of explained variance', fontsize=18)
        plt.title(f"{model_name} weighted bar map bottom", fontsize=18)

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
    pdf_path = os.path.join(result_dir, f"{model_name}_coords.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            rf_size = rf_size[0]
            # limits=(-rf_size//2, rf_size//2)
            limits = (-20, 20)

            plt.figure(figsize=(10,10))
            plt.suptitle(f"Estimations of {model_name} {layer_name} RF center coordinates using different techniques (n = {num_units_total})", fontsize=14)

            xdata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,2,1)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.ylabel('y')
            plt.title(f'ground truth top (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,2,3)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'ground truth bot (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (oc_t_df.FXVAR > fxvar_thres), 'MUX']
            ydata = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (oc_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,2,2)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.ylabel('y')
            plt.title(f'occluder top (n = {num_units_included})')
            ax = plt.gca()
            ax.invert_yaxis()

            xdata = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (oc_b_df.FXVAR > fxvar_thres), 'MUX']
            ydata = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (oc_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(xdata)
            plt.subplot(2,2,4)
            config_plot(limits)
            plt.scatter(xdata, ydata, alpha=0.4)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'occluder bot (n = {num_units_included})')
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

            plt.figure(figsize=(10,10))
            plt.suptitle(f"Estimations of {model_name} {layer_name} RF radii using different techniques (n = {num_units_total}, ERF = {rf_size})", fontsize=14)

            sd1data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,2,1)
            plt.hist(radii, bins=bins)
            plt.ylabel('counts')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'ground truth top (n = {num_units_included})')

            sd1data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,2,3)
            plt.hist(radii, bins=bins)
            plt.xlabel('$\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel('counts')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'ground truth bottom (n = {num_units_included})')

            sd1data = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (oc_t_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (oc_t_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,2,2)
            plt.hist(radii, bins=bins)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'occluder top (n = {num_units_included})')

            sd1data = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (oc_b_df.FXVAR > fxvar_thres), 'SD1']
            sd2data = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (oc_b_df.FXVAR > fxvar_thres), 'SD2']
            num_units_included = len(sd1data)
            radii = geo_mean(sd1data, sd2data)
            plt.subplot(2,2,4)
            plt.hist(radii, bins=bins)
            plt.xlabel('$\sqrt{sd_1^2+sd_2^2}$')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f'occluder bottom (n = {num_units_included})')
            
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
    pdf_path = os.path.join(result_dir, f"{model_name}_ori.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            ylim = (0, 3)

            plt.figure(figsize=(10,11))
            plt.suptitle(f"Estimations of {model_name} {layer_name} RF orientation using different techniques\n(n = {num_units_total})", fontsize=12)

            layer_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(221, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'ground truth top (n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(223, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'ground truth bottom (n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (oc_t_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(222, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'occluder map (top, n = {num_units_included})')
            plt.text(5, 0.2, 'eccentricity')

            layer_data = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (oc_b_df.FXVAR > fxvar_thres)]
            num_units_included = len(layer_data)
            ecc = eccentricity(layer_data['SD1'], layer_data['SD2'])
            ax = plt.subplot(224, projection='polar')
            ax.scatter(layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            ax.scatter(-layer_data['ORI']*math.pi/180, ecc, alpha=0.4)
            plt.ylim(ylim)
            plt.title(f'occluder map (bottom, n = {num_units_included})')
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
    pdf_path = os.path.join(result_dir, f"{model_name}_error_coords.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            limits = (-20, 20)
            
            plt.figure(figsize=(10, 10))
            plt.suptitle(f"Comparing RF center coordinates of different techniques of {model_name} {layer_name} (n = {num_units_total})", fontsize=14)

            gt_xdata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'MUX']
            ot_xdata = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'MUX']

            if sum(np.isfinite(gt_xdata)) == 0 or sum(np.isfinite(ot_xdata)) == 0: 
                continue  # Skip this layer if no data

            num_units_included = len(gt_xdata)
            plt.subplot(2,2,1)
            config_plot(limits)
            plt.scatter(gt_xdata, ot_xdata, alpha=0.4)
            plt.xlabel('GT x')
            plt.ylabel('Occluder x')
            r_val, p_val = pearsonr(gt_xdata, ot_xdata)
            plt.title(f'GT vs. occluder (top, n={num_units_included}, r={r_val:.2f})')

            gt_ydata = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'MUY']
            ot_ydata = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(gt_ydata)
            plt.subplot(2,2,2)
            config_plot(limits)
            plt.scatter(gt_ydata, ot_ydata, alpha=0.4)
            plt.xlabel('GT y')
            plt.ylabel('Occluder y')
            r_val, p_val = pearsonr(gt_ydata, ot_ydata)
            plt.title(f'GT vs. occluder (top, n={num_units_included}, r={r_val:.2f})')
            
            gb_xdata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'MUX']
            ob_xdata = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'MUX']
            num_units_included = len(gb_xdata)
            plt.subplot(2,2,3)
            config_plot(limits)
            plt.scatter(gb_xdata, ob_xdata, alpha=0.4)
            plt.xlabel('GT x')
            plt.ylabel('Occluder x')
            r_val, p_val = pearsonr(gb_xdata, ob_xdata)
            plt.title(f'GT vs. occluder (bottom, n={num_units_included}, r={r_val:.2f})')

            gb_ydata = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'MUY']
            ob_ydata = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'MUY']
            num_units_included = len(gb_ydata)
            plt.subplot(2,2,4)
            config_plot(limits)
            plt.scatter(gb_ydata, ob_ydata, alpha=0.4)
            plt.xlabel('GT y')
            plt.ylabel('Occluder y')
            r_val, p_val = pearsonr(gb_ydata, ob_ydata)
            plt.title(f'GT vs. occluder (bottom, n={num_units_included}, r={r_val:.2f})')

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

def make_error_radius_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_error_radius.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            limits = (0, 60)

            plt.figure(figsize=(10,5))
            plt.suptitle(f"Comparing RF radii of different techniques of {model_name} {layer_name} (n = {num_units_total}, ERF = {rf_size[0]})", fontsize=16)
            
            gt_sd1 = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'SD1']
            gt_sd2 = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'SD2']
            gt_radius = geo_mean(gt_sd1, gt_sd2)
            ot_sd1 = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'SD1']
            ot_sd2 = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres), 'SD2']
            ot_radius = geo_mean(ot_sd1, ot_sd2)
            num_top_units = len(gt_radius)

            if sum(np.isfinite(gt_radius)) == 0: 
                continue  # Skip this layer if no data

            plt.subplot(1,2,1)
            config_plot(limits)
            plt.scatter(gt_radius, ot_radius, alpha=0.4)
            plt.xlabel('GT $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel('Occluder $\sqrt{sd_1^2+sd_2^2}$')
            try:
                r_val, p_val = pearsonr(gt_radius, ot_radius)
            except:
                r_val = np.NaN
            plt.title(f'GT vs. Occluder (top, n={num_top_units}, r={r_val:.2f})')

            
            gb_sd1 = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'SD1']
            gb_sd2 = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'SD2']
            gb_radius = geo_mean(gb_sd1, gb_sd2)
            ob_sd1 = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'SD1']
            ob_sd2 = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres), 'SD2']
            ob_radius = geo_mean(ob_sd1, ob_sd2)
            num_bot_units = len(gb_radius)
            plt.subplot(1,2,2)
            config_plot(limits)
            plt.scatter(gb_radius, ob_radius, alpha=0.4)
            plt.xlabel('GT $\sqrt{sd_1^2+sd_2^2}$')
            plt.ylabel('Occluder $\sqrt{sd_1^2+sd_2^2}$')
            try:
                r_val, p_val = pearsonr(gb_radius, ob_radius)
            except:
                r_val = np.NaN
            plt.title(f'GT vs. Occluder (bottom, n={num_bot_units}, r={r_val:.2f})')

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
    # plt.ylim([0.75, 4])
    plt.xlabel('$\Delta \Theta $ (°)')

def delta_ori(ori_1, ori_2):
    # Note: this function assumes 0 <= ori < 180.
    theta_small = np.minimum(ori_1, ori_2)
    theta_large = np.maximum(ori_1, ori_2)
    # Because angles wraps around 0 and 180 deg, we need to consider two cases:
    delta_theta_a = theta_large - theta_small
    delta_theta_b = (theta_small + 180) - theta_large
    return np.minimum(delta_theta_a, delta_theta_b)

def make_error_ori_pdf():
    pdf_path = os.path.join(result_dir, f"{model_name}_error_ori.pdf")
    with PdfPages(pdf_path) as pdf:
        for conv_i, rf_size in enumerate(rf_sizes):
            # Get some layer-specific information.
            layer_name = f'conv{conv_i+1}'
            num_units_total = len(gt_t_df.loc[(gt_t_df.LAYER == layer_name)])
            
            # Get ground truth data (top and bottom)
            gt_data = gt_t_df.loc[(gt_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres)]
            gt_ecc = eccentricity(gt_data['SD1'], gt_data['SD2'])
            gt_ori = gt_data['ORI']
            gb_data = gt_b_df.loc[(gt_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres)]
            gb_ecc = eccentricity(gb_data['SD1'], gb_data['SD2'])
            gb_ori = gb_data['ORI']
            
            # Get weighted maps data (top and bottom)
            oc_t_data = oc_t_df.loc[(oc_t_df.LAYER == layer_name) & (gt_t_df.FXVAR > fxvar_thres) & (oc_t_df.FXVAR > fxvar_thres)]
            oc_t_ecc = eccentricity(oc_t_data['SD1'], oc_t_data['SD2'])
            oc_t_ori = oc_t_data['ORI']
            oc_b_data = oc_b_df.loc[(oc_b_df.LAYER == layer_name) & (gt_b_df.FXVAR > fxvar_thres) & (oc_b_df.FXVAR > fxvar_thres)]
            oc_b_ecc = eccentricity(oc_b_data['SD1'], oc_b_data['SD2'])
            oc_b_ori = oc_b_data['ORI']

            plt.figure(figsize=(10,11))
            plt.suptitle(f"Comparing {model_name} {layer_name} RF orientations of different techniques\n(n = {num_units_total})", fontsize=16)

            plt.subplot(2,2,1)
            plt.scatter(delta_ori(gt_ori, oc_t_ori), gt_ecc, alpha=0.4)
            config_plot()
            plt.ylabel('GT eccentricity')
            plt.title(f'GT vs. Occluder (top, n = {len(gt_ori)})')
            
            plt.subplot(2,2,2)
            plt.scatter(delta_ori(gt_ori, oc_t_ori), oc_t_ecc, alpha=0.4)
            config_plot()
            plt.ylabel('Occluder eccentricity')
            plt.title(f'GT vs. Occluder (top, n = {len(gt_ori)})')
            
            plt.subplot(2,2,3)
            plt.scatter(delta_ori(gb_ori, oc_b_ori), gb_ecc, alpha=0.4)
            config_plot()
            plt.ylabel('GT eccentricity')
            plt.title(f'GT vs. Occluder (bottom, n = {len(gb_ori)})')
            
            plt.subplot(2,2,4)
            plt.scatter(delta_ori(gb_ori, oc_b_ori), oc_b_ecc, alpha=0.4)
            config_plot()
            plt.ylabel('Occluder eccentricity')
            plt.title(f'GT vs. Occluder (bottom, n = {len(gb_ori)})')

            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    make_error_ori_pdf()
    pass

