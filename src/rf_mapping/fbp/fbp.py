"""
Script to calculate Fbp (Fraciton of bar part response)

Tony Fu, April 20, 2023
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import scipy.signal.windows as windows
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../../..')
from src.rf_mapping.bar import stimfr_bar
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.spatial import calculate_center
import src.rf_mapping.constants as c
from src.rf_mapping.result_txt_format import Rfmp4aSplist as Splist, CenterReponses as CR

model_name = 'alexnet'
layer_name = 'conv3'
unit_idx = 0
rank = 0

layer_idx = 6
model = models.alexnet(pretrained=True).to(c.DEVICE)
truncated_model = get_truncated_model(model, layer_idx)

result_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'mapping', model_name)
max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top5000_responses.txt")
splist = os.path.join(result_dir, f"{layer_name}_splist.txt")

center_responses = pd.read_csv(max_center_reponses_path, sep=' ', header=None)
center_responses.columns = [e.name for e in CR]

splist = pd.read_csv(splist, sep=' ', header=None)
splist.columns = [e.name for e in Splist]

cr_df = center_responses.loc[(center_responses['UNIT'] == unit_idx) & (center_responses['RANK'] == rank), ['STIM_I', 'R']]
stim_i = cr_df['STIM_I'].item()
original_response = cr_df['R'].item()

params = splist[splist['STIM_I'] == stim_i]

new_bar = stimfr_bar(params['XN'].item(), params['YN'].item(),
                     params['X0'].item(), params['Y0'].item(),
                     params['THETA'].item(), params['LEN'].item(), params['WID'].item(), 
                     params['AA'].item(), params['FGVAL'].item(), params['BGVAL'].item())

def blur_region(image, box, sigma):
    # y_min, x_min, y_max, x_max = box
    # blurred_image = gaussian_filter(image, sigma)
    # fused_image = image.copy()
    # fused_image[y_min:y_max+1, x_min:x_max+1] = blurred_image[y_min:y_max+1, x_min:x_max+1]
    # return fused_image

    # create a 2D Gaussian window with standard deviation of 3
    y_min, x_min, y_max, x_max = box
    mask = np.ones(image.shape)
    mask[y_min:y_max+1, x_min:x_max+1] = 0
    gaussian_mask = gaussian_filter(mask, sigma)
    image = image.copy() * gaussian_mask
    return image
    

plt.imshow(new_bar, cmap='gray')
plt.show()


new_bar = blur_region(new_bar, (60, 60, 80, 70), 5)
plt.imshow(new_bar, cmap='gray')
plt.show()


new_bar_color = np.stack([new_bar] * 3, axis=0)
bar_batch = np.expand_dims(new_bar_color, axis=0)

with torch.no_grad():  # turn off gradient calculations for speed.
    y = truncated_model(torch.tensor(bar_batch).type('torch.FloatTensor').to(c.DEVICE))
yc, xc = calculate_center(y.shape[-2:])
new_response = y[0, unit_idx, yc, xc].detach().cpu().numpy()

print(f"original response = {original_response}, new response = {new_response}")

