"""
Script to generate center of mass stats for the non-overlapping sums of the top
and bottom bars.

Tony Fu, July 21, 2022
"""
import os
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches


sys.path.append('..')
import constants as c


model_name = 'alexnet'

# Source paths:
stat_dir = os.path.join(c.REPO_DIR, 'results', 'rfmp4a', 'gaussian_fit', model_name)
txt_path = os.path.join(stat_dir, f"non_overlap.txt")

with open(txt_path) as f:
    lines = f.readlines()
    # Each line is made of: [layer_name unit num_max_bars num_min_bars]
    for line in lines:
        layer_name = line.split(' ')[0]
