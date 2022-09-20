"""
Make a pdf of the top N natural images and the gradient maps.

Tony Fu, June 26, 2022
"""
import os
import sys

import numpy as np
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.image import preprocess_img_for_plot, make_box
from src.rf_mapping.spatial import SpatialIndexConverter
from src.rf_mapping.guided_backprop import GuidedBackprop
from src.rf_mapping.reproducibility import set_seeds
import src.rf_mapping.constants as c

# Please specify some details here:
# set_seeds()
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = "alexnet"
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
top_n = 5
image_size = (227, 227)
is_random = False

# Please double-check the directories:
img_dir = c.IMG_DIR

if is_random:
    index_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth',
                             'top_n_random', model_name)
else:
    index_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth',
                              'top_n', model_name)
result_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth',
                          'top_n', 'test')

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take hours to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Initiate helper objects.
converter = SpatialIndexConverter(model, image_size)
conv_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, nums_units = conv_counter.count()


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


for conv_i, layer_idx in enumerate(layer_indices):
    if conv_i != 1: continue
    top_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=True)
    bot_grad_method = GuidedBackprop(model, layer_idx, remove_neg_grad=False)
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indices = np.load(index_path).astype(int)
    # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]

    num_units = nums_units[conv_i]
    print(f"Making pdf for {layer_name}...")

    pdf_path = os.path.join(result_dir, f"{layer_name}.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, plt_axes = plt.subplots(4, top_n)
        fig.set_size_inches(20, 15)
        
        # Collect axis and imshow handles in a list.
        ax_handles = []
        im_handles = []
        for ax_row in plt_axes:
            for ax in ax_row:
                ax_handles.append(ax)
                im_handles.append(ax.imshow(np.zeros((*image_size, 3))))

        for unit_i in tqdm(range(num_units)):
            fig.suptitle(f"{layer_name} unit no.{unit_i}", fontsize=20)
            # Get top and bottom image indices and patch spatial indices
            max_n_img_indices   = max_min_indices[unit_i, :top_n, 0]
            max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]
            min_n_img_indices   = max_min_indices[unit_i, :top_n, 2]
            min_n_patch_indices = max_min_indices[unit_i, :top_n, 3]

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

            # Bottom N images and gradient patches:
            for i, (min_img_idx, min_patch_idx) in enumerate(zip(min_n_img_indices,
                                                            min_n_patch_indices)):
                box = converter.convert(min_patch_idx, layer_idx, 0, is_forward=False)
                plot_one_img(im_handles[i+2*top_n], ax_handles[i+2*top_n], min_img_idx, box)
                ax_handles[i+2*top_n].set_title(f"bottom {i+1} image")

                plot_one_grad_map(im_handles[i+3*top_n], ax_handles[i+3*top_n],
                                  min_img_idx, unit_i, min_patch_idx,
                                  box, bot_grad_method)
                ax_handles[i+3*top_n].set_title(f"bottom {i+1} gradient")

            pdf.savefig(fig)
            plt.close()
