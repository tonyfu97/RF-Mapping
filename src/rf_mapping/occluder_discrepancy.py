"""
The visualization method is inspired by paper "Object Detectors Emerge in Deep
Scene CNNs (2015)" by Zhou et al., Avaliable at: https://arxiv.org/abs/1412.6856

In the paper's section 3.2 "VISUALIZING THE RECEPTIVE FIELDS OF UNITS AND THEIR
ACTIVATION PATTERNS," the authors described their method:

    "... we replicate each image many times with small random occluders (image
    patches of size 11x11) at different locations in the image. Specifically,
    we generate occluders in a dense grid with a stride of 3. This results in
    about 5000 occluded images per original image. We now feed all the occluded
    images into the same network and record the change in activation as
    compared to using the original image. If there is a large discrepancy, we
    know that the given patch is important and vice versa. This allows us to
    build a discrepancy map for each image. Finally, to consolidate the
    information from the K images, we center the discrepancy map around the
    spatial location of the unit that caused the maximum activation for the
    given image. Then we average the re-centered discrepancy maps to generate
    the final RF."

Their code is avaliable at:
https://github.com/zhoubolei/cnnvisualizer/blob/master/pytorch_generate_unitsegments.py

The following Python script also uses occluders to generate discrepancy maps.
However, unlike Zhou et al. (2015), the occluders were only used on the top
and bottom image patches, rather than entire images. Also, instead of using
a fixed size and stride for the occluders, we scaled the occluder size and
stride linearly with the RF field size according to the formula:

    occluder_size = rf_size // 10
    occluder_stride = occluder_size // 3
    
This method will serve as an alternative to the guided backprop visualizations
for the 'ground truth' RF mapping.

Tony Fu, Aug 5, 2022
"""
import os
import sys
from unittest import result

import numpy as np
import torch
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.image import preprocess_img_for_plot, make_box, preprocess_img_to_tensor
from src.rf_mapping.spatial import SpatialIndexConverter, get_rf_sizes
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c


#######################################.#######################################
#                                                                             #
#                                DRAW_OCCLUDER                                #
#                                                                             #
###############################################################################
def draw_occluder(img_tensor, top_left, bottom_right):
    """
    Returns an occluded version of img without modifying the original img.
    Note: the y-axis points downward.

    Parameters
    ----------
    img : torch.tensor
        Image with pixel values comparable to [-1, 1] and with the color
        channel as the first dimension.
    top_left : (int, int)
        Spatial index of the top left corner (inclusive) of the occluder.
    bottom_right : (int, int))
        Spatial index of the bottom right corner (inclusive) of the occluder.

    Returns
    -------
    occluded_img_tensor : torch.tensor
        img but with random occluder at the specified location.
    """
    occluded_img_tensor = img_tensor.clone().detach()
    occluder_size = (bottom_right[0] - top_left[0] + 1,
                     bottom_right[1] - top_left[1] + 1)
    occluder = (torch.rand(3, *occluder_size).to(c.DEVICE) * 2) - 1
    occluded_img_tensor[0, :, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = occluder
    return occluded_img_tensor[0]


# Plot an example occluder.
if __name__ == "__main__":
    img = torch.zeros((1, 3, 5, 10))
    occluded_img = draw_occluder(img, (0, 3), (3, 7))
    plt.imshow(np.transpose(occluded_img.numpy(), (1,2,0)))


#######################################.#######################################
#                                                                             #
#                               OCCLUDER_PARMS                                #
#                                                                             #
###############################################################################
def get_occluder_params(box, rf_size, image_size):
    """
    Given an image, generate the random occluders in the patch, which is
    bounded by the box.
    
    Parameters
    ----------
    box : (int, int, int, int)
        Spatial index of the patch to place the occluders in. Note that the
        occluders are going to be much smaller than this box.
    rf_size : int
        Side length of RF (assumed to be square). Sometimes, the box may be
        cropped, so we will use rf_size instead of the size of the 'box' to
        calculate the occluder size and stride.
    
    Returns
    -------
    occluder_params : [{str : (int, int), str : (int, int)}, ...]
        List of dictionary, which contains the coodinates of 'top_left' and
        'bottom_right' corners (inclusive, in (y, x)) format. Note: the y-axis
        points downward.
    """
    occluder_size = max(rf_size // 10, 1)
    occluder_stride = max(occluder_size // 3, 1)
    occluder_params = []

    vx_min, hx_min, vx_max, hx_max = box
    
    if (vx_max - vx_min + 1 > rf_size) or (hx_max - hx_min + 1 > rf_size):
        raise ValueError("The box should not be larger than RF_size.")

    for i in np.arange(vx_min, max(vx_max-occluder_size+2, vx_min+1), occluder_stride):
        for j in np.arange(hx_min, max(hx_max-occluder_size+2, hx_min+1), occluder_stride):
            # Skip if indexing out of range:
            if (i+occluder_size-1 > image_size[0]-1) or (j+occluder_size-1 > image_size[1]-1):
                continue

            occluder_params.append({'top_left' : (i, j), 
                                    'bottom_right' : (i+occluder_size-1, j+occluder_size-1)})
    return occluder_params


# Plot the occulder boxes.
if __name__ == "__main__":
    box = (10, 19, 20, 29)
    occluder_params = get_occluder_params(box, 20, (227,227))
    print(len(occluder_params))
    plt.ylim([5, 25])
    plt.xlim([15, 35])
    plt.grid()
    ax = plt.gca()
    for occluder_param in occluder_params:
        occ_box = make_box((occluder_param['top_left'][0],
                            occluder_param['top_left'][1],
                            occluder_param['bottom_right'][0],
                            occluder_param['bottom_right'][1],))
        ax.add_patch(occ_box)
    plt.show()


#######################################.#######################################
#                                                                             #
#                             ADD_DISCREPANCY_MAPS                            #
#                                                                             #
###############################################################################
def add_discrepancy_maps(response_diff, occluder_params, discrepancy_maps, box,
                         rf_size):
    """
    Weighs the occluding area by the absolute difference between i-th unit's
    response to the original image and the occluded image. This function
    modifies the discrepancy_maps[unit] in-place.

    Parameters
    ----------
    response_diff : numpy.ndarray [occluders_batch_size]
        Response difference between original image and occluded images.
    occluder_params : [{str : (int, int), str : (int, int)}, ...]
        See the Returns of get_occluder_params().
    discrepancy_map : numpy.ndarray [num_units, rf_size, rf_size]
        The sum of occluder area weighted by the absolute difference between
        of the response to original image and the occluded image.
    box : (int, int, int, int)
        Spatial index of the patch to place the occluders in. Note that the
        occluders are going to be much smaller than this box.
    rf_size : int
        The theoretical side length of the square RF.
    """
    vx_min, hx_min, vx_max, hx_max = box

    # define the vertical and horizontal offset of the occluding box such that
    # the box is between [0, rf_size - 1].
    if vx_min == 0:
        v_offset = (vx_max - vx_min + 1) - rf_size + vx_min
        # The offset is negative when the left side of the box is cropped off.
    else:
        v_offset = vx_min
    if hx_min == 0:
        h_offset = (hx_max - hx_min + 1) - rf_size + hx_min
        # The offset is negative when the left side of the box is cropped off.
    else:
        h_offset = hx_min

    for occluder_i, occluder_param in enumerate(occluder_params):
        occ_vx_min, occ_hx_min = occluder_param['top_left']
        occ_vx_max, occ_hx_max = occluder_param['bottom_right']
        
        # translate occluder params (spatial indices) to be between [0, rf_size-1].
        occ_vx_min -= v_offset
        occ_hx_min -= h_offset
        occ_vx_max -= v_offset
        occ_hx_max -= h_offset
        
        discrepancy_maps[occ_vx_min:occ_vx_max+1, occ_hx_min:occ_hx_max+1] +=\
                                              np.abs(response_diff[occluder_i])


#######################################.#######################################
#                                                                             #
#                              GET_DISCREPANCY_MAP                            #
#                                                                             #
###############################################################################
def get_discrepancy_map(img, occluder_params, truncated_model, rf_size,
            spatial_index, unit_i, batch_size=100, _debug=False, image_size=(227,227)):
    """
    Presents the occluded image and returns the discrepency maps (one for
    each unit in the final layer of the truncated model).

    Parameters
    ----------
    img : numpy.ndarray
        The image.
    occluder_params : [{str : (int, int), str : (int, int)}, ...]
        See the Returns of get_occluder_params().
    truncated_model : UNION[fx.graph_module.GraphModule, torch.nn.Module]
        Truncated model. The feature maps of the last layer will be used to
        create the discrepancy maps.
    rf_size : int
        The side length of the receptive field, which is assumed to be squared.
    batch_size : int, default = 100
        How many occluded images to present at once.
    _debug : bool, default = False
        If true, only present the first few occluded images.

    Returns
    -------
    discrepancy_map : numpy.ndarray [rf_size, rf_size]
        The sum of occluder area weighted by the absolute difference between
        of the response to original image and the occluded image.
    """
    img_tensor = preprocess_img_to_tensor(img)
    y = truncated_model(img_tensor)
    yc, xc = np.unravel_index(spatial_index, y.shape[-2:])
    original_response = y[:, unit_i, yc, xc].cpu().detach().numpy()

    occluder_i = 0
    num_stim = len(occluder_params)
    discrepancy_map = np.zeros((rf_size, rf_size))

    while (occluder_i < num_stim):
        sys.stdout.write('\r')
        sys.stdout.write(f"Presenting {occluder_i}/{num_stim} stimuli...")
        sys.stdout.flush()

        real_batch_size = min(batch_size, num_stim-occluder_i)
        occluder_batch = torch.zeros((real_batch_size, 3, *image_size)).to(c.DEVICE)

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = occluder_params[occluder_i + i]
            occluder_batch[i] = draw_occluder(img_tensor,
                                              params['top_left'],
                                              params['bottom_right'])

        # Present the patch of bars to the truncated model.
        y = truncated_model(occluder_batch)
        y = y.cpu().detach().numpy()
        yc, xc = np.unravel_index(spatial_index, y.shape[-2:])
        response_diff = y[:, unit_i, yc, xc] - original_response
        if this_is_a_test_run:
            # Skip some occluders for testing.
            add_discrepancy_maps(response_diff,
                                occluder_params[occluder_i:occluder_i+batch_size:5], discrepancy_map, box, rf_size)
        else:
            add_discrepancy_maps(response_diff,
                             occluder_params[occluder_i:occluder_i+batch_size], discrepancy_map, box, rf_size)
        occluder_i += real_batch_size

    return discrepancy_map


# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = "alexnet"
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
top_n = 5
image_size = (227, 227)
this_is_a_test_run = False
batch_size = 200

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
        if conv_i == 0: 
            continue
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
                                                          max_patch_idx, unit_i, batch_size=batch_size, _debug=this_is_a_test_run, image_size=image_size)
                    im_handles[i].set_data(discrepancy_map/discrepancy_map.max())
                    ax_handles[i].set_title(f"top {i+1} image")
                    max_discrepancy_maps[i] = discrepancy_map

                # Bottom N images and gradient patches:
                for i, (min_img_idx, min_patch_idx) in enumerate(zip(min_n_img_indices,
                                                                min_n_patch_indices)):
                    img_path = os.path.join(img_dir, f"{min_img_idx}.npy")
                    img = np.load(img_path)
                    box = converter.convert(min_patch_idx, layer_idx, 0, is_forward=False)
                    occluder_params = get_occluder_params(box, rf_size, image_size)
                    discrepancy_map = get_discrepancy_map(img, occluder_params, 
                                                          truncated_model, rf_size,
                                                          min_patch_idx, unit_i, batch_size=batch_size, _debug=this_is_a_test_run, image_size=image_size)
                    im_handles[i+top_n].set_data(discrepancy_map/discrepancy_map.max())
                    ax_handles[i+top_n].set_title(f"bottom {i+1} image")
                    min_discrepancy_maps[i] = discrepancy_map

                plt.show()
                pdf.savefig(fig)
                plt.close()
        
        # Save the results in npy files.
        max_map_path = os.path.join(result_dir, f'{layer_name}_max.npy')
        min_map_path = os.path.join(result_dir, f'{layer_name}_min.npy')
        np.save(max_map_path, max_discrepancy_maps)
        np.save(min_map_path, min_discrepancy_maps)
