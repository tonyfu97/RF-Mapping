"""
A motivational figure that illustrates the need of RF mapping--See how
spurious borderownership selectivity arises when we use the maximal RF.

Tony Fu, October 12th, 2022
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.image import make_box
from src.rf_mapping.spatial import (xn_to_center_rf,
                                    calculate_center,
                                    get_rf_sizes,)
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c
from src.rf_mapping.stimulus import *
from src.rf_mapping.bar import stimfr_bar_color

# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True)
# model_name = 'resnet18'
conv_i = 1
unit_i = 132

# Please specify the output directory
result_dir = os.path.join(c.REPO_DIR, 'results', 'figures')

# Get the model info
xn_list = xn_to_center_rf(model, image_size=(999, 999))  
layer_indicies, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
layer_index = layer_indicies[conv_i]
truncated_model = get_truncated_model(model, layer_index)

# Find the bar sizes
xn = xn_list[conv_i]
max_rf = max_rfs[conv_i][0]
padding = (xn - max_rf) // 2

###############################################################################

def make_bar_set(xn, x0, x_offset, y0, y_offset, theta,
                 blen, bwid, aa, color1, color2):
    """Create the 4 bars."""
    bar1 = stimfr_bar_color(xn,xn,x0-x_offset,y0-y_offset,theta,blen,bwid,aa,*color1, *color2)
    bar2 = stimfr_bar_color(xn,xn,x0+x_offset,y0+y_offset,theta,blen,bwid,aa,*color2, *color1)
    bar3 = stimfr_bar_color(xn,xn,x0-x_offset,y0-y_offset,theta,blen,bwid,aa,*color2, *color1)
    bar4= stimfr_bar_color(xn,xn,x0+x_offset,y0+y_offset,theta,blen,bwid,aa,*color1, *color2)
    return [bar1, bar2, bar3, bar4]


def make_bar_set_tensor(bar_set):
    """Collect bars into a single torch tensor batch."""
    bar_batch = torch.zeros((len(bar_set), 3, xn, xn))
    for i, bar in enumerate(bar_set):
        bar_batch[i] = torch.tensor(bar)
    return bar_batch
    

def get_center_responses(bar_set_tensor, truncated_model):
    """Get the center responses."""
    y = truncated_model(bar_set_tensor)
    y_center = (y.shape[-1] - 1) // 2
    return y[:, unit_i, y_center, y_center].detach().numpy()


###############################################################################

# Define bar parameters
color1 = (-1, -1, -1)
color2 = (1, 1, 0.8)
theta = 0
blen = 40
bwid = 20
aa = 0.5
x0 = 0
y0 = 0
x_offset = bwid/2 * math.cos(theta/180 * math.pi)
y_offset = bwid/2 * math.sin(theta/180 * math.pi)

# Make bars and get center responses
bar_set = make_bar_set(xn, x0, x_offset, y0, y_offset, theta,
                       blen, bwid, aa, color1, color2)
bar_set_tensor = make_bar_set_tensor(bar_set)
center_responses = get_center_responses(bar_set_tensor, truncated_model)

###############################################################################

# Define bar parameters
color1 = (-1, -1, -1)
color2 = (1, 1, 0.8)
theta = 0
# blen = 18
# bwid = 9
# aa = 0.5
# x0 = -15
# y0 = 0
blen = 8
bwid = 4
aa = 0.5
x0 = -3.6
y0 = -0.8
x_offset = bwid/2 * math.cos(theta/180 * math.pi)
y_offset = bwid/2 * math.sin(theta/180 * math.pi)


# Make bars and get center responses
fix_bar_set = make_bar_set(xn, x0, x_offset, y0, y_offset, theta,
                           blen, bwid, aa, color1, color2)
fix_bar_set_tensor = make_bar_set_tensor(fix_bar_set)
fix_center_responses = get_center_responses(fix_bar_set_tensor, truncated_model)

###############################################################################

# Make the pdf
pdf_path = os.path.join(result_dir, f"motivation_{model_name}_conv{conv_i+1}_{unit_i}.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(20, 5))
    for i, bar in enumerate(bar_set):
        plt.subplot(1, 4, i+1)
        plt.imshow(np.transpose(bar, (1,2,0))/2 + 0.5)
        plt.gca().add_patch(make_box((padding, padding, xn - padding - 2, xn - padding - 2)))
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{i+1}", fontsize=20)

    pdf.savefig()
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(4), center_responses)
    plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    plt.hlines(0, -1, 4, colors=('black'))
    plt.xlabel('input image', fontsize=20)
    plt.ylabel('response', fontsize=20)
    plt.title(f'Conv{conv_i+1}-{unit_i} Response', fontsize=20)
    
    pdf.savefig()
    plt.show()
    plt.close()
    
    
    plt.figure(figsize=(20, 5))
    for i, bar in enumerate(fix_bar_set):
        plt.subplot(1, 4, i+1)
        plt.imshow(np.transpose(bar, (1,2,0))/2 + 0.5)
        plt.gca().add_patch(make_box((padding, padding, xn - padding - 2, xn - padding - 2)))
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{i+1}", fontsize=20)

    pdf.savefig()
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(4), fix_center_responses)
    plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    plt.hlines(0, -1, 4, colors=('black'))
    plt.xlabel('input image', fontsize=20)
    plt.ylabel('response', fontsize=20)
    plt.title(f'Conv{conv_i+1}-{unit_i} Response', fontsize=20)
    
    pdf.savefig()
    plt.show()
    plt.close()
