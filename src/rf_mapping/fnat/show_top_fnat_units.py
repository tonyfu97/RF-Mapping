"""
To plot the natural images of the units that has a large or small FNAT.

Tony Fu, Sep 26, 2022
"""
import os
import sys

import numpy as np
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import SpatialIndexConverter
from src.rf_mapping.image import preprocess_img_for_plot, make_box
from src.rf_mapping.guided_backprop import GuidedBackprop


# Please specify the model
model = models.alexnet()
model_name = 'alexnet'
# model = models.vgg16()
# model_name = 'vgg16'
# model = models.resnet18()
# model_name = 'resnet18'

this_is_a_test_run = False
map1_name = 'gt'                # ['gt', 'occlude']
map2_name = 'rfmp4a'            # ['rfmp4a', 'rfmp4c7o', 'rfmp_sin1', 'pasu']
fnat_upper_thres = 0.8
fnat_lower_thres = 0.05
image_size = (227, 227)
top_n = 5

# Please double-check the directories:
img_dir = c.IMG_DIR
index_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', model_name)
fnat_dir = os.path.join(c.REPO_DIR, 'results', 'fnat', map2_name, model_name)
result_dir = fnat_dir

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n]") 
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Initiate helper objects.
converter = SpatialIndexConverter(model, image_size)
conv_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, nums_units = conv_counter.count()


############################  HELPER FUNCTIONS ################################


def plot_one_img(im, ax, img_idx, box):
    """Plots the image and draw the red box."""
    # Remove patches
    try:
        ax.patches.pop()
    except:
        pass
    # Plot image
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    img = preprocess_img_for_plot(img)
    im.set_data(img)
    # Remove tick marks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.add_patch(make_box(box))


def plot_one_grad_map(im, ax, img_idx, unit_idx, patch_idx, box, grad_method):
    """Plots the target unit's gradient map for the image."""
    # Remove patches
    try:
        ax.patches.pop()
    except:
        pass
    # Plot gradients
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp_map = grad_method.generate_gradients(img, unit_idx, patch_idx)
    im.set_data(preprocess_img_for_plot(gbp_map))
    # Remove tick marks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.add_patch(make_box(box))
    
    
##############################  LOAD FNAT DATA  ###############################


# Load the fnat
top_n_r = 10
fnat_path = os.path.join(fnat_dir, f"{map2_name}_fnat_{top_n_r}_avg.txt")
fnat_df = pd.read_csv(fnat_path, sep=" ", header=None)
fnat_df.columns = ['LAYER', 'UNIT', 'TOP_AVG_FNAT', 'BOT_AVG_FNAT']


#################################  MAKE PDF  ##################################


# PDF 1. Visualize those with large FNAT.
pdf_path = os.path.join(result_dir, f"{model_name}_{map2_name}_fnat_greater_than_{fnat_upper_thres}.pdf")
with PdfPages(pdf_path) as pdf:
    for conv_i, layer_idx in enumerate(layer_indices):
        # Skipping Conv1
        if conv_i == 0:
            continue

        top_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=True)
        bot_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=False)
        layer_name = f"conv{conv_i + 1}"
        
        # Load top image indices
        index_path = os.path.join(index_dir, f"{layer_name}.npy")
        max_min_indices = np.load(index_path).astype(int)
        # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]
        
        # Extract CRI for this layer
        top_cri_df = fnat_df.loc[(fnat_df.LAYER == layer_name) & (fnat_df.TOP_AVG_FNAT > fnat_upper_thres), ['UNIT', 'TOP_AVG_FNAT']]
        bot_cri_df = fnat_df.loc[(fnat_df.LAYER == layer_name) & (fnat_df.BOT_AVG_FNAT > fnat_upper_thres), ['UNIT', 'BOT_AVG_FNAT']]

        top_num_units = len(top_cri_df.UNIT)
        top_num_units = len(bot_cri_df.UNIT)
        print(f"Making pdf for {layer_name}...")

        fig, plt_axes = plt.subplots(2, top_n)
        fig.set_size_inches(20, 8)
        
        # Collect axis and imshow handles in a list.
        ax_handles = []
        im_handles = []
        for ax_row in plt_axes:
            for ax in ax_row:
                ax_handles.append(ax)
                im_handles.append(ax.imshow(np.zeros((*image_size, 3))))

        # Plot the natural images of the top units.
        for _, unit_data in tqdm(top_cri_df.iterrows()):
            unit_i = int(unit_data['UNIT'])
            this_fnat = unit_data['TOP_AVG_FNAT']
            
            fig.suptitle(f"{layer_name} unit no.{unit_i} (fnat = {this_fnat})", fontsize=20)
            # Get top and bottom image indices and patch spatial indices
            max_n_img_indices   = max_min_indices[unit_i, :top_n, 0]
            max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]

            # Top N images and gradient patches:
            for i, (max_img_idx, max_patch_idx) in enumerate(zip(max_n_img_indices,
                                                              max_n_patch_indices)):
                box = converter.convert(max_patch_idx, layer_idx, 0, is_forward=False)
                plot_one_img(im_handles[i], ax_handles[i], max_img_idx, box)
                ax_handles[i].set_title(f"top {i+1} image")

                plot_one_grad_map(im_handles[i+top_n], ax_handles[i+top_n],
                                  max_img_idx, unit_i, max_patch_idx,
                                  box, top_grad_method)
                ax_handles[i+top_n].set_title(f"top {i+1} gradient")

            pdf.savefig(fig)
            plt.close()

        # TODO: show the natural images of the bottom units as well.


# PDF 2. Visualize those with small FNAT.
pdf_path = os.path.join(result_dir, f"{model_name}_{map2_name}_fnat_less_than_{fnat_lower_thres}.pdf")
with PdfPages(pdf_path) as pdf:
    for conv_i, layer_idx in enumerate(layer_indices):
        # Skipping Conv1
        if conv_i == 0:
            continue

        top_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=True)
        bot_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=False)
        layer_name = f"conv{conv_i + 1}"
        
        # Load top image indices
        index_path = os.path.join(index_dir, f"{layer_name}.npy")
        max_min_indices = np.load(index_path).astype(int)
        # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]
        
        # Extract CRI for this layer
        top_cri_df = fnat_df.loc[(fnat_df.LAYER == layer_name) & (fnat_df.TOP_AVG_FNAT < fnat_lower_thres), ['UNIT', 'TOP_AVG_FNAT']]
        bot_cri_df = fnat_df.loc[(fnat_df.LAYER == layer_name) & (fnat_df.BOT_AVG_FNAT < fnat_lower_thres), ['UNIT', 'BOT_AVG_FNAT']]

        top_num_units = len(top_cri_df.UNIT)
        top_num_units = len(bot_cri_df.UNIT)
        print(f"Making pdf for {layer_name}...")

        fig, plt_axes = plt.subplots(2, top_n)
        fig.set_size_inches(20, 8)
        
        # Collect axis and imshow handles in a list.
        ax_handles = []
        im_handles = []
        for ax_row in plt_axes:
            for ax in ax_row:
                ax_handles.append(ax)
                im_handles.append(ax.imshow(np.zeros((*image_size, 3))))

        # Plot the natural images of the top units.
        for _, unit_data in tqdm(top_cri_df.iterrows()):
            unit_i = int(unit_data['UNIT'])
            this_fnat = unit_data['TOP_AVG_FNAT']
            
            fig.suptitle(f"{layer_name} unit no.{unit_i} (fnat = {this_fnat})", fontsize=20)
            # Get top and bottom image indices and patch spatial indices
            max_n_img_indices   = max_min_indices[unit_i, :top_n, 0]
            max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]

            # Top N images and gradient patches:
            for i, (max_img_idx, max_patch_idx) in enumerate(zip(max_n_img_indices,
                                                              max_n_patch_indices)):
                box = converter.convert(max_patch_idx, layer_idx, 0, is_forward=False)
                plot_one_img(im_handles[i], ax_handles[i], max_img_idx, box)
                ax_handles[i].set_title(f"top {i+1} image")

                plot_one_grad_map(im_handles[i+top_n], ax_handles[i+top_n],
                                  max_img_idx, unit_i, max_patch_idx,
                                  box, top_grad_method)
                ax_handles[i+top_n].set_title(f"top {i+1} gradient")

            pdf.savefig(fig)
            plt.close()

        # TODO: show the natural images of the bottom units as well.
        