"""
Make a pdf of the sum of the gradient patches.

Tony Fu, June 30, 2022
"""
import os
import sys

import numpy as np
from pathlib import Path
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from hook import ConvUnitCounter
from image import preprocess_img_for_plot


# Please specify some details here:
model = models.alexnet()
model_name = "alexnet"
sum_modes = ['abs', 'sqr']

# Please double-check the directories:
backprop_sum_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/ground_truth/backprop_sum/{model_name}')
pdf_dir = backprop_sum_dir

###############################################################################

# Get info of conv layers.
conv_counter = ConvUnitCounter(model)
layer_indicies, nums_units = conv_counter.count()
num_layers = len(layer_indicies)

for sum_mode in sum_modes:
    backprop_sum_dir_with_modes = os.path.join(backprop_sum_dir, sum_mode)
    pdf_dir_with_sum_modes = backprop_sum_dir_with_modes

    for conv_i in range(num_layers):
        layer_name = f"conv{conv_i + 1}"
        num_units = nums_units[conv_i]

        print(f"Making pdf for {model_name} conv{conv_i + 1} sum mode: {sum_mode}")
        pdf_path = os.path.join(pdf_dir_with_sum_modes, f"{layer_name}.pdf")
        with PdfPages(pdf_path) as pdf:

            for unit_i in tqdm(range(5)): #num_units)):
                max_sum_path = os.path.join(backprop_sum_dir_with_modes, f"max_conv{conv_i+1}.{unit_i}.npy")
                min_sum_path = os.path.join(backprop_sum_dir_with_modes, f"min_conv{conv_i+1}.{unit_i}.npy")
                both_sum_path = os.path.join(backprop_sum_dir_with_modes, f"both_conv{conv_i+1}.{unit_i}.npy")

                max_sum = np.load(max_sum_path)
                min_sum = np.load(min_sum_path)
                both_sum = np.load(both_sum_path)

                plt.figure(figsize=(15,5))
                plt.suptitle(f"conv{conv_i+1} unit no.{unit_i} (sum mode = {sum_mode})", fontsize=24)
                plt.subplot(1, 3, 1)
                plt.imshow(preprocess_img_for_plot(max_sum), cmap='gray')
                plt.title("max", fontsize=20)
                plt.subplot(1, 3, 2)
                plt.imshow(preprocess_img_for_plot(min_sum), cmap='gray')
                plt.title("min", fontsize=20)
                plt.subplot(1, 3, 3)
                plt.imshow(preprocess_img_for_plot(both_sum), cmap='gray')
                plt.title("max + min", fontsize=20)

                pdf.savefig()
                plt.close
