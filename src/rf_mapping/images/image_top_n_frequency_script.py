"""
Code to show to the images that are most frequently rank as the top- or bottom-
N images.

Tony Fu, August 29 2022
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
from src.rf_mapping.image import preprocess_img_for_plot, make_box, preprocess_img_to_tensor
from src.rf_mapping.spatial import SpatialIndexConverter
from src.rf_mapping.guided_backprop import GuidedBackprop
import src.rf_mapping.constants as c

# Please specify some details here:
# model = models.alexnet().to(c.DEVICE)
# model_name = 'alexnet'
# model = models.vgg16().to(c.DEVICE)
# model_name = "vgg16"
model = models.resnet18().to(c.DEVICE)
model_name = "resnet18"

image_size = (227, 227)
top_n = 5

# Please double-check the directories:
img_dir = c.IMG_DIR
rank_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', model_name)
result_dir = rank_dir

# Get info of conv layers.
converter = SpatialIndexConverter(model, image_size)
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
num_layers = len(layer_indices)

###############################################################################
#                                                                             #
#                          PDF NO.1 MOST COMMON IMAGES                        #
#                                                                             #
###############################################################################
def norm_img(img):
    return (img - img.min()) / (img.max() - img.min())

pdf_path = os.path.join(result_dir, f"most_common_images.pdf")
with PdfPages(pdf_path) as pdf:
    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i+1}"
        rankings = np.load(os.path.join(rank_dir, f"{layer_name}.npy"))
        # [num_units, top_n, 4]
        # The 4 columns: top_n_img_idx, top_n_img_pos, bot_n_img_idx, bot_n_img_pos
        
        top_n_img_idx = rankings[:, :top_n, 0].flatten()
        top_unique, top_counts = np.unique(top_n_img_idx, return_counts=True)
        sorted_top_counts = np.sort(top_counts)[::-1]
        most_frequent_top_sorted_descending = top_unique[np.argsort(top_counts)[::-1]]
        
        plt_idx = 1
        plt.figure(figsize=(top_n*5, 11))
        plt.suptitle(f"Most common top and bottom-{top_n} images in {model_name} {layer_name}", fontsize=24)
        for img_idx, count in zip(most_frequent_top_sorted_descending[:top_n],
                                  sorted_top_counts[:top_n]):
            img = np.load(os.path.join(img_dir, f"{img_idx}.npy"))
            img_tensor = preprocess_img_to_tensor(img)
            img = norm_img(img)
            plt.subplot(2, top_n, plt_idx)
            plt.imshow(np.transpose(img, (1,2,0)))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"img no.{img_idx} (top, count = {count})\ntensor: mean = {img_tensor.mean():.4f}, std = {img_tensor.std():.4f}", fontsize=12)
            plt_idx += 1
        
        bot_n_img_idx = rankings[:, :top_n, 2].flatten()
        bot_unique, bot_counts = np.unique(bot_n_img_idx, return_counts=True)
        sorted_bot_counts = np.sort(bot_counts)[::-1]
        most_frequent_bot_sorted_descending = bot_unique[np.argsort(bot_counts)[::-1]]
        
        for img_idx, count in zip(most_frequent_bot_sorted_descending[:top_n],
                                  sorted_bot_counts[:top_n]):
            img = np.load(os.path.join(img_dir, f"{img_idx}.npy"))
            img_tensor = preprocess_img_to_tensor(img)
            img = norm_img(img)
            plt.subplot(2, top_n, plt_idx)
            plt.imshow(np.transpose(img, (1,2,0)))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"img no.{img_idx} (bottom, count = {count})\ntensor: mean = {img_tensor.mean():.4f}, std = {img_tensor.std():.4f}", fontsize=12)
            plt_idx += 1

        pdf.savefig()
        plt.close()

###############################################################################
#                                                                             #
#                       PDF NO.2 MOST COMMON IMAGE PATCHES                    #
#                                                                             #
###############################################################################
def most_common_img_patches(img_idx, img_pos, num_patches):
    """
    Finds the most common images patches. Returns them as an ordered dictionary
    with the most common (img_idx, img_pos) pair as the first key.
    """
    counts = {}
    for pair in zip(img_idx, img_pos):
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    sorted_counts = {pair: count for pair, count in sorted(counts.items(), key=lambda item: item[1])[::-1][:num_patches]}
    return sorted_counts

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
    ax.add_patch(make_box(box, linewidth=2))
        

pdf_path = os.path.join(result_dir, f"most_common_image_patches.pdf")
num_patches = 10
with PdfPages(pdf_path) as pdf:
    for conv_i, layer_idx in enumerate(layer_indices):
        layer_name = f"conv{conv_i+1}"
        rankings = np.load(os.path.join(rank_dir, f"{layer_name}.npy"))
        # [num_units, top_n, 4]
        # The 4 columns: top_n_img_idx, top_n_img_pos, bot_n_img_idx, bot_n_img_pos
        
        top_n_img_idx = rankings[:, :top_n, 0].flatten()
        top_n_img_pos = rankings[:, :top_n, 1].flatten()
        top_counts = most_common_img_patches(top_n_img_idx, top_n_img_pos, num_patches)
        
        bot_n_img_idx = rankings[:, :top_n, 2].flatten()
        bot_n_img_pos = rankings[:, :top_n, 3].flatten()
        bot_counts = most_common_img_patches(bot_n_img_idx, bot_n_img_pos, num_patches)
        
        fig, plt_axes = plt.subplots(2, num_patches)
        fig.set_size_inches(num_patches*5, 11)
        fig.suptitle(f"Most common top and bottom-{top_n} images in {model_name} {layer_name}", fontsize=24)
        
        # Collect axis and imshow handles in a list.
        ax_handles = []
        im_handles = []
        for ax_row in plt_axes:
            for ax in ax_row:
                ax_handles.append(ax)
                im_handles.append(ax.imshow(np.zeros((*image_size, 3))))

        plt_idx = 0
        for (img_idx, img_pos), count in top_counts.items():
            box = converter.convert(img_pos, layer_idx, 0, is_forward=False)
            plot_one_img(im_handles[plt_idx], ax_handles[plt_idx], img_idx, box)
            ax_handles[plt_idx].set_title(f"img no.{img_idx} pos.{img_pos} (top, count = {count})")
            plt_idx += 1

        for (img_idx, img_pos), count in bot_counts.items():
            box = converter.convert(img_pos, layer_idx, 0, is_forward=False)
            plot_one_img(im_handles[plt_idx], ax_handles[plt_idx], img_idx, box)
            ax_handles[plt_idx].set_title(f"img no.{img_idx} pos.{img_pos} (bottom, count = {count})")
            plt_idx += 1

        pdf.savefig()
        plt.close()
