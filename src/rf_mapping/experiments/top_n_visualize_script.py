"""
Visualize the top N natural images and the guided backpropation by making a
pdf for each layer.

Tony Fu, June 26, 2022
"""
import os

import numpy as np
from pathlib import Path
from torchvision import models
import matplotlib.pyplot as plt

from hook import SpatialIndexConverter
from image import preprocess_img_for_plot, make_box
from guided_backprop import GuidedBackprop

# Please specify some details here:
model = models.alexnet(pretrained=True)
model_name = "alexnet"
top_n = 5

# Please double-check the directories:
img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"
index_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/top_n/{model_name}')
result_dir = index_dir

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take hours to run. Are you sure? ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

converter = SpatialIndexConverter(model, (227, 227))

layer_name = "conv2"
layer_idx = 3
index_path = os.path.join(index_dir, f"{layer_name}.npy")

max_min_indicies = np.load(index_path).astype(int)  # [units * top_n_img * [max_img_idx, max_idx, min_img_idx, min_idx]]

unit_idx = 0

plt.figure(figsize=(20, 7), facecolor='w')
plt.suptitle(f"{layer_name} unit no.{unit_idx}", fontsize=20)
for i, img_idx in enumerate(max_min_indicies[unit_idx, :top_n, 0]):
    plt.subplot(2, top_n, i+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    img = preprocess_img_for_plot(img)
    plt.imshow(img)
    plt.title(f"top {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = max_min_indicies[unit_idx, i, 1]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))

for i, img_idx in enumerate(max_min_indicies[unit_idx, :top_n, 2]):
    plt.subplot(2, top_n, i+top_n+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    img = preprocess_img_for_plot(img)
    plt.imshow(img)
    plt.title(f"bottom {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = max_min_indicies[unit_idx, i, 3]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))
plt.show()


plt.figure(figsize=(20, 7), facecolor='w')
plt.suptitle(f"{layer_name} unit no.{unit_idx}", fontsize=20)
gbp = GuidedBackprop(model)
for i, img_idx in enumerate(max_min_indicies[unit_idx, :top_n, 0]):
    plt.subplot(2, top_n, i+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp_map = gbp.generate_gradients(img, layer_idx, unit_idx, max_min_indicies[unit_idx, :top_n, 1])
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.title(f"top {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = max_min_indicies[unit_idx, i, 1]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))


for i, img_idx in enumerate(max_min_indicies[unit_idx, :top_n, 2]):
    plt.subplot(2, top_n, i+top_n+1)
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    gbp = GuidedBackprop(model)
    gbp_map = gbp.generate_gradients(img, layer_idx, unit_idx, max_min_indicies[unit_idx, :top_n, 3])
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.title(f"top {i+1} image")
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    patch_index = max_min_indicies[unit_idx, i, 3]
    box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
    ax.add_patch(make_box(box_indicies))
