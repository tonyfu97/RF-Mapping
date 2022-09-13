"""
Plot the shapes in descending order of its responses.

Note: all code assumes that the y-axis points downward.

Tony Fu, September 12th, 2022
"""
import os
import sys

from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.pasu_shape import make_pasu_shape
from src.rf_mapping.spatial import get_num_layers
from src.rf_mapping.result_txt_format import (CenterReponses as CR,
                                              PasuBWSplist as SP,)

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
this_is_a_test_run = False
top_n = 10

# Please double-check the directories:
if this_is_a_test_run:
    source_dir = os.path.join(c.REPO_DIR, 'results', 'pasu', 'mapping', 'test')
else:
    source_dir = os.path.join(c.REPO_DIR, 'results', 'pasu', 'mapping', model_name)
    
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'pasu', 'analysis', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'pasu', 'analysis', model_name)
    

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else:
#         raise KeyboardInterrupt("Interrupted by user")

num_layers = get_num_layers(model)


# Define some helper functions
def set_column_names(df, Format):
    """Name the columns of the pandas DF according to Format."""
    df.columns = [e.name for e in Format]


def plot_one_shape(stim_i, im, ax):
    params = splist_df.loc[splist_df.STIM_I == stim_i]
    shape = make_pasu_shape(params.XN.item(), params.YN.item(),
                            params.X0.item(), params.Y0.item(),
                            params.SI.item(), params.RI.item(),
                            params.FGVAL.item(), params.BGVAL.item(),
                            params.SIZE.item(), plot=False)
    im.set_data(shape)
    # Remove tick marks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])


for conv_i in range(num_layers):
    layer_name = f"conv{conv_i+1}"
    
    # Load the splist and center responses.
    splist_path = os.path.join(source_dir, f"{layer_name}_splist.txt")
    top_cr_path = os.path.join(source_dir, f"{layer_name}_top5000_responses.txt")
    bot_cr_path = os.path.join(source_dir, f"{layer_name}_bot5000_responses.txt")
    
    try:
        splist_df = pd.read_csv(splist_path, sep=" ", header=None)
        top_cr_df = pd.read_csv(top_cr_path, sep=" ", header=None)
        bot_cr_df = pd.read_csv(bot_cr_path, sep=" ", header=None)
    except:
        continue
    
    set_column_names(splist_df, SP)
    set_column_names(top_cr_df, CR)
    set_column_names(bot_cr_df, CR)
    
    num_units = len(top_cr_df['UNIT'].unique())
    map_size = splist_df.loc[0, 'XN']

    pdf_path = os.path.join(result_dir, f"{model_name}_{layer_name}_top_bot_shapes.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, plt_axes = plt.subplots(2, top_n)
        fig.set_size_inches(top_n * 2, 5)
        
        # Collect axis and imshow handles in a list.
        ax_handles = []
        im_handles = []
        for ax_row in plt_axes:
            for ax in ax_row:
                ax_handles.append(ax)
                im_handles.append(ax.imshow(np.zeros((map_size, map_size)),
                                            vmin=-1, vmax=1, cmap='gray'))
        
        for unit_i in tqdm(range(num_units)):
            fig.suptitle(f"{model_name} {layer_name} no.{unit_i} top/bottom Pasupathy shapes", fontsize=18)

            subplot_i = 0

            # Top-N shapes
            for rank_i in range(top_n):
                stim_i = top_cr_df.loc[(top_cr_df.UNIT == unit_i) & (top_cr_df.RANK == rank_i), 'STIM_I'].item()
                plot_one_shape(stim_i, im_handles[subplot_i], ax_handles[subplot_i])
                ax_handles[subplot_i].set_title(f"top {rank_i+1}\nshape no.{stim_i}")
                subplot_i += 1

            # Bottom-N shapes
            for rank_i in range(top_n):
                stim_i = bot_cr_df.loc[(bot_cr_df.UNIT == unit_i) & (bot_cr_df.RANK == rank_i), 'STIM_I'].item()
                plot_one_shape(stim_i, im_handles[subplot_i], ax_handles[subplot_i])
                ax_handles[subplot_i].set_title(f"bottom {rank_i+1}\nshape no.{stim_i}")
                subplot_i += 1

            pdf.savefig(fig)
            plt.close()
