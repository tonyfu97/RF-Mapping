import os


import numpy as np
from pathlib import Path
import torch
from torchvision import models
import matplotlib.pyplot as plt
%matplotlib inline


from hook import SizeInspector
from spatial_index import SpatialIndexConverter
from image import preprocess_img_for_plot, make_box, one_sided_zero_pad
from guided_backprop import GuidedBackprop


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained = True).to(device)
model_name = "alexnet"
result_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/{model_name}')
converter = SpatialIndexConverter(model, (227, 227))

layer_name = "conv2"
layer_idx = 3
result_path = os.path.join(result_dir, f"{layer_name}.npy")

conv_results = np.load(result_path).astype(int)  # [units * top_n_img * [max_img_idx, max_idx, min_img_idx, min_idx]]


top_n = 5
unit_idx = 0
img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"

plt.figure(figsize=(20, 7), facecolor='w')
plt.suptitle(f"{layer_name} unit no.{unit_idx}", fontsize=20)
gbp = GuidedBackprop(model)
for i, img_idx in enumerate(conv_results[unit_idx, :top_n, 0]):
    plt.subplot(2, top_n, i+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp_map = gbp.generate_gradients(img, layer_idx, unit_idx, conv_results[unit_idx, :top_n, 1])
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.title(f"top {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = conv_results[unit_idx, i, 1]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))
    

for i, img_idx in enumerate(conv_results[unit_idx, :top_n, 2]):
    plt.subplot(2, top_n, i+top_n+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp = GuidedBackprop(model)
    gbp_map = gbp.generate_gradients(img, layer_idx, unit_idx, conv_results[unit_idx, :top_n, 3])
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.title(f"top {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = conv_results[unit_idx, i, 3]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))
