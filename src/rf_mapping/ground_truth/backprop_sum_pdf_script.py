"""
Make a pdf of the sum of the gradient patches.

Tony Fu, June 30, 2022

Converted to matplotlib-efficient version on July 25. NOT TESTED YET.
"""
import os
import sys

import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.image import preprocess_img_for_plot
import src.rf_mapping.constants as c


# Please specify some details here:
model = models.alexnet()
model_name = "alexnet"
sum_modes = ['abs', 'sqr']
this_is_a_test_run = False

# Please double-check the directories:
backprop_sum_dir = c.REPO_DIR + f'/results/ground_truth/backprop_sum/{model_name}'
pdf_dir = backprop_sum_dir

###############################################################################

# Get info of conv layers.
conv_counter = ConvUnitCounter(model)
layer_indices, nums_units = conv_counter.count()
num_layers = len(layer_indices)

for sum_mode in sum_modes:
    backprop_sum_dir_with_modes = os.path.join(backprop_sum_dir, sum_mode)
    pdf_dir_with_sum_modes = backprop_sum_dir_with_modes

    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i + 1}"
        num_units = nums_units[conv_i]
        
        # Load maps.
        max_sum_path = os.path.join(backprop_sum_dir_with_modes, f"{layer_name}_max.npy")
        min_sum_path = os.path.join(backprop_sum_dir_with_modes, f"{layer_name}_min.npy")
        max_sum = np.load(max_sum_path)
        min_sum = np.load(min_sum_path)
        both_sum = (max_sum + min_sum)/2

        print(f"Making pdf for {model_name} conv{conv_i + 1} sum mode: {sum_mode}")
        pdf_path = os.path.join(pdf_dir_with_sum_modes, f"{layer_name}.pdf")
        with PdfPages(pdf_path) as pdf:
            _, yn, xn = max_sum.shape

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(15, 5)
            im1 = ax1.imshow(np.zeros((yn, xn)), cmap='gray', vmin=0, vmax=1)
            im2 = ax2.imshow(np.zeros((yn, xn)), cmap='gray', vmin=0, vmax=1)

            for unit_i in tqdm(range(num_units)):
                # Do only the first 5 unit during testing phase
                if this_is_a_test_run and unit_i >= 5:
                    break

                fig.suptitle(f"Gradient average of image patches "
                             f"({layer_name} no.{unit_i}, "
                             f"sum mode: {sum_mode})", fontsize=20)

                im1.set_data(preprocess_img_for_plot(max_sum[unit_i]))
                ax1.set_title("max", fontsize=16)

                im2.set_data(preprocess_img_for_plot(min_sum[unit_i]))
                ax2.set_title("min", fontsize=16)

                pdf.savefig(fig)
                plt.close
