import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import src.rf_mapping.constants as c

MODEL_NAME = "alexnet"
LAYER_NAME = "conv3"
FNAT_DIR = f"{c.RESULTS_DIR}/rfmp4a/fnat/{MODEL_NAME}/{LAYER_NAME}"

############################  SOME HELPER FUNCTIONS ###########################

TOP_IDX = 0
BOT_IDX = 1

def load_df(test_name):
    fnat_path = os.path.join(FNAT_DIR, f"{test_name}_fnat.txt")
    df = pd.read_csv(fnat_path, sep='\s+')
    return df


def calculate_fnat_diff(df, default_index):
    param_col = df.columns[0]
    default_df = df[df[param_col] == default_index]
    num_params = len(df[param_col].unique())
    
    num_stim_list = []
    top_diff_list = []
    bot_diff_list = []
    
    for param_i in range(num_params):
        param_df = df[df[param_col] == param_i]
        num_stim_list.append(param_df.num_stim.values[0])
        top_diff_list.append(param_df['top_fnat'].values  - default_df['top_fnat'].values)
        bot_diff_list.append(param_df['bot_fnat'].values  - default_df['bot_fnat'].values)
    
    return num_params, num_stim_list, top_diff_list, bot_diff_list


def plot_fnat_bar(test_name, num_params, top_diff_list, bot_diff_list, ylim=[-0.15, 0.05], pdf=None):
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"FNAT Difference from Default\n{MODEL_NAME} {LAYER_NAME} {test_name}", fontsize=16)
    
    plt.subplot(2, 1, 1)
    plt.boxplot(top_diff_list)
    plt.hlines(0, 1, num_params, colors='k', linestyles='dashed')
    plt.ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    plt.title('Top', fontsize=14)
    # plt.ylim(ylim)
    
    plt.subplot(2, 1, 2)
    plt.boxplot(bot_diff_list)
    plt.hlines(0, 1, num_params, colors='k', linestyles='dashed')
    plt.ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    plt.title('Bottom', fontsize=14)
    # plt.ylim(ylim)

    if pdf is not None:
        pdf.savefig()
    plt.show()
    

def plot_fnat_bar2(test_name, num_params, top_diff_list, bot_diff_list, df, num_stim_list, ylim=[-0.15, 0.05], pdf=None):
    param_col = df.columns[0]
    avg_top_fnat = df.groupby(param_col).mean().top_fnat
    avg_bot_fnat = df.groupby(param_col).mean().bot_fnat
    ax1_xval = np.array(num_stim_list)  # now ax1_xval are values from num_stim_list

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f"FNAT Difference from Default\n{MODEL_NAME} {LAYER_NAME} {test_name}", fontsize=16)

    # Subplot 1
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(ax1_xval, avg_top_fnat, 'o-')
    ax1.set_ylabel('Average FNAT', fontsize=14)
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_xticklabels(ax1_xval, rotation=90)

    # create individual boxplots at each location in ax1_xval
    for i in range(len(ax1_xval)):
        ax2.boxplot(top_diff_list[i], positions=[ax1_xval[i]])
    ax2.hlines(0, min(ax1_xval)-1000, max(ax1_xval)+1000, colors='k', linestyles='dashed')
    ax2.set_xlabel('Number of stimuli', fontsize=14)
    ax2.set_ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    ax2.set_ylim([-0.5, 0.5])

    # Subplot 2
    ax1 = axs[1]
    ax2 = ax1.twinx()
    ax1.plot(ax1_xval, avg_bot_fnat, 'o-')
    ax1.set_xlabel('Number of stimuli', fontsize=14)
    ax1.set_ylabel('Average FNAT', fontsize=14)
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_xticklabels(ax1_xval, rotation=90)


    # create individual boxplots at each location in ax1_xval
    for i in range(len(ax1_xval)):
        ax2.boxplot(bot_diff_list[i], positions=[ax1_xval[i]])
    ax2.hlines(0, min(ax1_xval)-1000, max(ax1_xval)+1000, colors='k', linestyles='dashed')
    ax2.set_xlabel('Number of stimuli', fontsize=14)
    ax2.set_ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    ax2.set_ylim([-0.5, 0.5])


    fig.tight_layout()

    if pdf is not None:
        pdf.savefig()
    plt.show()


    

def plot_fnat_vs_num_stim(test_name, num_stim_list, top_diff_list, bot_diff_list, ylim=[-0.15, 0.05], pdf=None):
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"FNAT Difference from Default\n{MODEL_NAME} {LAYER_NAME} {test_name}", fontsize=16)
    
    plt.subplot(2, 1, 1)
    top_diff_avg = [np.mean(e) for e in top_diff_list]
    plt.plot(num_stim_list, top_diff_avg, 'o-')
    plt.hlines(0, min(num_stim_list), max(num_stim_list), colors='k', linestyles='dashed')
    plt.xlabel('Number of stimuli', fontsize=14)
    plt.ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    plt.title('Top', fontsize=14)
    plt.ylim(ylim)
    
    plt.subplot(2, 1, 2)
    bot_diff_avg = [np.mean(e) for e in bot_diff_list]
    plt.plot(num_stim_list, bot_diff_avg, 'o-')
    plt.hlines(0, min(num_stim_list), max(num_stim_list), colors='k', linestyles='dashed')
    plt.xlabel('Number of stimuli', fontsize=14)
    plt.ylabel('Average FNAT Difference\n(From Default)', fontsize=14)
    plt.title('Bottom', fontsize=14)
    plt.ylim(ylim)

    if pdf is not None:
        pdf.savefig()
    plt.show()

    
###############################################################################

if __name__ == "__main__":
    # Due to bad software design, I have to manually specify the test names and
    # the corresponding default index for each test. Please see rfmp4a_fnat_script.py
    # for the default index for each test.
    # Default index is the index of the default parameter in the rfmp4a paradigm.
    tests = {
        'bar_length': 4,
        'aspect_ratio': 3,
        'orientations': 3,
        'grid_divider': 1,
    }
    
    for test_name, default_index in tests.items():
        df = load_df(test_name)
        num_params, num_stim_list, top_diff_list, bot_diff_list = calculate_fnat_diff(df, default_index=default_index)
        
        pdf_path = os.path.join(FNAT_DIR, f"{MODEL_NAME}_{LAYER_NAME}_{test_name}_fnat_diff.pdf")
        with PdfPages(pdf_path) as pdf:
            # plot_fnat_bar(test_name, num_params, top_diff_list, bot_diff_list, ylim=[-0.15, 0.05], pdf=pdf)
            # plot_fnat_vs_num_stim(test_name, num_stim_list, top_diff_list, bot_diff_list, ylim=[-0.15, 0.05], pdf=pdf)
            plot_fnat_bar2(test_name, num_params, top_diff_list, bot_diff_list, df, num_stim_list, pdf=pdf)
