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
from src.rf_mapping.spatial import SpatialIndexConverter, get_rf_sizes
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c
from src.rf_mapping.occluder_discrepancy import (get_occluder_params,
                                                 get_discrepancy_map)


# Please specify some details here:
# model = models.alexnet(pretrained=True).to(c.DEVICE)
# model_name = "alexnet"
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
model = models.resnet18(pretrained=True).to(c.DEVICE)
model_name = "resnet18"
top_n = 5
image_size = (227, 227)
this_is_a_test_run = False
batch_size = 10

# Please double-check the directories:
img_dir = c.IMG_DIR
index_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n', model_name)
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'occlude', 'mapping', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'occlude', 'mapping', model_name)

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take hours to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

# Initiate helper objects.
converter = SpatialIndexConverter(model, image_size)
conv_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, nums_units = conv_counter.count()
layer_indices, rf_sizes = get_rf_sizes(model, image_size)

if __name__ == "__main__":
    for conv_i, layer_idx in enumerate(layer_indices):
        truncated_model = get_truncated_model(model, layer_idx)
        layer_name = f"conv{conv_i + 1}"
        index_path = os.path.join(index_dir, f"{layer_name}.npy")
        max_min_indices = np.load(index_path).astype(int)
        # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]
        
        rf_size = rf_sizes[conv_i][0]
        num_units = nums_units[conv_i]
        
        # Array intializations
        max_discrepancy_maps = np.zeros((top_n, num_units, rf_size, rf_size))
        min_discrepancy_maps = np.zeros((top_n, num_units, rf_size, rf_size))

        pdf_path = os.path.join(result_dir, f"{layer_name}.pdf")
        with PdfPages(pdf_path) as pdf:
            fig, plt_axes = plt.subplots(2, top_n)
            fig.set_size_inches(20, 8)
            
            # Collect axis and imshow handles in a list.
            ax_handles = []
            im_handles = []
            for ax_row in plt_axes:
                for ax in ax_row:
                    ax_handles.append(ax)
                    im_handles.append(ax.imshow(np.zeros((rf_size, rf_size)),
                                                vmin=0, vmax=1, cmap='gray'))

            for unit_i in tqdm(range(num_units)):
                if this_is_a_test_run and unit_i > 2:
                    continue
                sys.stdout.write('\r')
                sys.stdout.write(f"Making pdf for {layer_name} no.{unit_i}...")
                sys.stdout.flush()
                print()

                fig.suptitle(f"{layer_name} unit no.{unit_i}", fontsize=20)
                # Get top and bottom image indices and patch spatial indices
                max_n_img_indices   = max_min_indices[unit_i, :top_n, 0]
                max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]
                min_n_img_indices   = max_min_indices[unit_i, :top_n, 2]
                min_n_patch_indices = max_min_indices[unit_i, :top_n, 3]

                # Top N images and gradient patches:
                for i, (max_img_idx, max_patch_idx) in enumerate(zip(max_n_img_indices,
                                                                max_n_patch_indices)):
                    img_path = os.path.join(img_dir, f"{max_img_idx}.npy")
                    img = np.load(img_path)
                    box = converter.convert(max_patch_idx, layer_idx, 0, is_forward=False)
                    occluder_params = get_occluder_params(box, rf_size, image_size)
                    discrepancy_map = get_discrepancy_map(img, occluder_params, 
                                                          truncated_model, rf_size,
                                                          max_patch_idx, unit_i, box,
                                                          batch_size=batch_size, _debug=this_is_a_test_run, image_size=image_size)
                    im_handles[i].set_data(discrepancy_map/discrepancy_map.max())
                    ax_handles[i].set_title(f"top {i+1} image")
                    max_discrepancy_maps[i][unit_i] = discrepancy_map

                # Bottom N images and gradient patches:
                for i, (min_img_idx, min_patch_idx) in enumerate(zip(min_n_img_indices,
                                                                min_n_patch_indices)):
                    img_path = os.path.join(img_dir, f"{min_img_idx}.npy")
                    img = np.load(img_path)
                    box = converter.convert(min_patch_idx, layer_idx, 0, is_forward=False)
                    occluder_params = get_occluder_params(box, rf_size, image_size)
                    discrepancy_map = get_discrepancy_map(img, occluder_params, 
                                                          truncated_model, rf_size,
                                                          min_patch_idx, unit_i, box,
                                                          batch_size=batch_size, _debug=this_is_a_test_run, image_size=image_size)
                    im_handles[i+top_n].set_data(discrepancy_map/discrepancy_map.max())
                    ax_handles[i+top_n].set_title(f"bottom {i+1} image")
                    min_discrepancy_maps[i][unit_i] = discrepancy_map

                plt.show()
                pdf.savefig(fig)
                plt.close()
        
        # Save the results in npy files.
        max_map_path = os.path.join(result_dir, f'{layer_name}_max.npy')
        min_map_path = os.path.join(result_dir, f'{layer_name}_min.npy')
        np.save(max_map_path, max_discrepancy_maps)
        np.save(min_map_path, min_discrepancy_maps)
