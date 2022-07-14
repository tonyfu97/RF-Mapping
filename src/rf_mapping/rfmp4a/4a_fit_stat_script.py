"""
Script to summarize the Gaussian fit statistics.

Tony Fu, July 14, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from gaussian_fit import gaussian_fit, ParamCleaner
from gaussian_fit import GaussianFitParamFormat as ParamFormat
from hook import ConvUnitCounter
from spatial import get_rf_sizes
from mapping import RfMapper
from image import make_box
import constants as c


# Please specify some details here:
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model_name = 'alexnet'
cumulate_modes = ['weighted', 'or']
image_shape = (227, 227)
this_is_a_test_run = False

# Please double-check the directories:
fit_stat_dir = c.REPO_DIR + f'/results/rfmp4a/gaussian_fit/{model_name}'
pdf_dir = fit_stat_dir

###############################################################################

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
_, rf_sizes = get_rf_sizes(model, image_shape, layer_type=nn.Conv2d)

# Helper objects:
param_cleaner = ParamCleaner()

for cumulate_mode in cumulate_modes:
    for max_or_min in ['max', 'min', 'both']:
        fit_stat_dir_with_mode = os.path.join(fit_stat_dir, cumulate_mode)
        pdf_dir_with_mode = os.path.join(pdf_dir, cumulate_mode)
        pdf_path = os.path.join(pdf_dir_with_mode, f"summary_{max_or_min}.pdf")
        num_layers = len(rf_sizes)

        with PdfPages(pdf_path) as pdf:
            for conv_i, rf_size in enumerate(tqdm(rf_sizes)):