"""
TODO:

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
from src.rf_mapping.net import get_truncated_model

# Please specify some details here:
model = models.alexnet().to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16().to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18().to(c.DEVICE)
# model_name = "resnet18"

image_size = (227, 227)
img_idx = 44011
layer_idx = 8
unit_idx = 0
pos_idx = 100

# Please double-check the directories:
img_dir = c.IMG_DIR
rank_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', model_name)
result_dir = rank_dir

# Get info of conv layers.
converter = SpatialIndexConverter(model, image_size)
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()
num_layers = len(layer_indices)

img = np.load(os.path.join(img_dir, f"{img_idx}.npy"))
img_tensor = preprocess_img_to_tensor(img)

truncated_model = get_truncated_model(model, layer_idx)
y = truncated_model(img_tensor)
feature_map = y[0, unit_idx].detach().numpy()
plt.imshow(feature_map)





