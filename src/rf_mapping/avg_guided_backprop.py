import os

import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
# %matplotlib inline

from hook import get_rf_sizes, SpatialIndexConverter
from image import one_sided_zero_pad, preprocess_img_for_plot
from guided_backprop import GuidedBackprop


# Script guard.
if __name__ == "__main__":
    user_input = input("This code takes time to run. Are you sure? "\
                       "Enter 'y' to proceed. Type any other key to stop: ")
    if user_input == 'y':
        sum_mode = input("Choose a summation mode: {'sum', 'abs', 'sqr', 'relu'}: ")
        double_check = input(f"All .npy files in the result_dir will be deleted. Are you sure? (y/n): ")
        if user_input == 'y':
            pass
        else:
            raise KeyboardInterrupt("Interrupted by user")
    else: 
        raise KeyboardInterrupt("Interrupted by user")


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained = True).to(device)
model_name = "alexnet"
index_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/indicies/{model_name}')

img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"
layer_indicies, rf_sizes = get_rf_sizes(model, (227, 227), nn.Conv2d)
sum_modes = {'sum', 'abs', 'sqr', 'relu'}
sum_mode = 'sqr'
result_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/backprop_sum/{model_name}/{sum_mode}')

def delete_all_npy_files(dir):
    for f in os.listdir(dir):
        if f.endswith('.npy'):
            os.remove(os.path.join(dir, f))
delete_all_npy_files(result_dir)

gbp = GuidedBackprop(model)
converter = SpatialIndexConverter(model, (227, 227))

for conv_i, rf_size in enumerate(rf_sizes):
    layer_idx = layer_indicies[conv_i]
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indicies = np.load(index_path).astype(int)
    num_units, num_images, _ = max_min_indicies.shape
    print(f"Summing guided backprop results for {layer_name}...")
 
    for unit_i in tqdm(range(num_units)):
        unit_max_sum = np.zeros((rf_size[0], rf_size[1], 3))
        unit_min_sum = np.zeros((rf_size[0], rf_size[1], 3))
        
        for img_i in range(num_images):
            # TODO: Do the same for min sum.
            max_img_idx, max_idx, min_img_idx, min_idx = max_min_indicies[unit_i, img_i, :]
            
            vx_min, hx_min, vx_max, hx_max = converter.convert(max_idx, layer_idx, 0, is_forward=False)
            img_path = os.path.join(img_dir, f"{img_i}.npy")
            img = np.load(img_path)
            gbp_map = gbp.generate_gradients(img, layer_idx, unit_i, max_idx)
            gbp_patch = gbp_map[:, vx_min:vx_max+1, hx_min:hx_max+1]
            gbp_patch_padded = one_sided_zero_pad(gbp_patch, rf_size, (vx_min, hx_min, vx_max, hx_max))
            
            if sum_mode == 'sum':
                unit_max_sum += gbp_patch_padded
            elif sum_mode == 'abs':
                unit_max_sum += np.absolute(gbp_patch_padded)
            elif sum_mode == 'sqr':
                unit_max_sum += np.square(gbp_patch_padded)
            elif sum_mode == 'relu':
                gbp_patch_padded[gbp_patch_padded<0] = 0
                unit_max_sum += gbp_patch_padded
            else:
                raise KeyError("sum mode must be 'abs', 'sqr', or 'relu'.")
            
        unit_max_sum_norm = unit_max_sum/num_units
        unit_min_sum_norm = unit_min_sum/num_units
        plt.figure(figsize=(8,15))
        plt.suptitle(f"conv{conv_i+1} unit no.{unit_i} sum mode: {sum_mode}")
        plt.subplot(1, 2, 1)
        plt.imshow(preprocess_img_for_plot(unit_max_sum_norm))
        plt.title("max")
        plt.subplot(1, 2, 2)
        plt.imshow(preprocess_img_for_plot(unit_min_sum_norm))
        plt.title("min")
        plt.show()
        
        print("Saving responses...")
        max_result_path = os.path.join(result_dir, f"max_conv{conv_i+1}.{unit_i}.npy")
        min_result_path = os.path.join(result_dir, f"min_conv{conv_i+1}.{unit_i}.npy")
        np.save(max_result_path, unit_max_sum_norm)
        np.save(min_result_path, unit_min_sum_norm)





# gbp = GuidedBackprop(model)
# for i, img_idx in enumerate(conv_results[unit_idx, :top_n, 0]):
#     plt.subplot(2, top_n, i+1)
#     img_path = os.path.join(img_dir, f"{img_idx}.npy")
#     img = np.load(img_path)
#     gbp_map = gbp.generate_gradients(img, layer_idx, unit_idx, conv_results[unit_idx, :top_n, 1])
#     plt.imshow(preprocess_img_for_plot(gbp_map))
#     plt.title(f"top {i+1} image")
#     plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
#     ax = plt.gca()
#     patch_index = conv_results[unit_idx, i, 1]
#     box_indicies = converter.convert(patch_index, layer_idx, 0, is_forward=False)
#     ax.add_patch(make_box(box_indicies))
    