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


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained = True).to(device)
model_name = "alexnet"
result_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/{model_name}')

img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"
layer_indicies, rf_sizes = get_rf_sizes(model, (227, 227), nn.Conv2d)
sum_modes = {'sum', 'abs', 'sqr', 'relu'}
sum_mode = 'abs'

gbp = GuidedBackprop(model)
converter = SpatialIndexConverter(model, (227, 227))


model_max_sums = []
model_min_sums = []
for conv_i, rf_size in enumerate(rf_sizes):
    if conv_i == 0 or conv_i == 1 or conv_i == 2:
        continue
    
    layer_idx = layer_indicies[conv_i]
    layer_name = f"conv{conv_i + 1}"
    result_path = os.path.join(result_dir, f"{layer_name}.npy")
    conv_results = np.load(result_path).astype(int)
    num_units, num_images, _ = conv_results.shape
    print(f"Summing guided backprop results for {layer_name}...")
    layer_max_sums = []
    layer_min_sums = []
 
    for unit_i in tqdm(range(num_units)):
        unit_max_sum = np.zeros((rf_size[0], rf_size[1], 3))
        unit_min_sum = np.zeros((rf_size[0], rf_size[1], 3))
        
        for img_i in range(num_images):
            max_img_idx, max_idx, min_img_idx, min_idx = conv_results[unit_i, img_i, :]
            
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
            
        layer_max_sums.append(unit_max_sum/num_units)
        layer_min_sums.append(unit_min_sum/num_units)
        print(preprocess_img_for_plot(layer_max_sums[-1]).shape)
        plt.imshow(preprocess_img_for_plot(layer_max_sums[-1]))
        plt.show()
    
    model_max_sums.append(layer_max_sums)
    model_min_sums.append(layer_min_sums)




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
    