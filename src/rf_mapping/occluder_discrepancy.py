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

Author: Tony Fu
Date Created: Aug 5, 2022
Last Modified: Aug 25, 2022

Update: Dr. Bair has pointed out that by adding the entire occluder areas to
the map, the algorithm has blurred the result as if it is convolved with a
uniform kernel of the same size. In order avoid blurring, Dr. Bair suggested
to only add the center of the occluders to the map. This change was made on
August 25, 2022.
"""
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append('../..')
from src.rf_mapping.image import make_box, preprocess_img_to_tensor
import src.rf_mapping.constants as c


#######################################.#######################################
#                                                                             #
#                            DRAW_RANDOM_OCCLUDER                             #
#                                                                             #
###############################################################################
def draw_random_occluder(img_tensor, top_left, bottom_right):
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
    bottom_right : (int, int)
        Spatial index of the bottom right corner (inclusive) of the occluder.

    Returns
    -------
    occluded_img_tensor : torch.tensor
        img but with random occluder at the specified location.
    """
    occluded_img_tensor = img_tensor.clone().detach()
    occluder_size = (bottom_right[0] - top_left[0] + 1,
                     bottom_right[1] - top_left[1] + 1)
    
    # Occluder is a random patch with uniform distribution over [-1, +1),
    # set independently for RGB color channels.
    occluder = (torch.rand(3, *occluder_size).to(c.DEVICE) * 2) - 1
    
    occluded_img_tensor[0, :, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = occluder
    return occluded_img_tensor[0]


# Plot an example occluder.
if __name__ == "__main__":
    img = torch.zeros((1, 3, 5, 10))
    occluded_img = draw_random_occluder(img, (0, 3), (3, 7))
    plt.imshow(np.transpose(occluded_img.numpy(), (1,2,0)))
    plt.show()



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
    occluder_stride = max(occluder_size // 2, 1)
    # occluder_size = 3
    # occluder_stride = 1
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
        
        # translate occluder params (spatial indices that spans the entire
        # image) to be between [0, rf_size-1].
        occ_vx_min -= v_offset
        occ_hx_min -= h_offset
        occ_vx_max -= v_offset
        occ_hx_max -= h_offset
        
        # discrepancy_maps[occ_vx_min:occ_vx_max+1, occ_hx_min:occ_hx_max+1] +=\
        #                                       np.abs(response_diff[occluder_i])
        
        # add the center of the occluder weighted by the abs(response).
        mid_vx = (occ_vx_max + occ_vx_min) // 2
        mid_hx = (occ_hx_max + occ_hx_min) // 2
        discrepancy_maps[mid_vx, mid_hx] += np.abs(response_diff[occluder_i])


#######################################.#######################################
#                                                                             #
#                          GET_RANDOM_DISCREPANCY_MAP                         #
#                                                                             #
###############################################################################
def get_random_discrepancy_map(img, occluder_params, truncated_model, rf_size,
                    spatial_index, unit_i, box, batch_size=100, _debug=False,
                    image_size=(227,227)):
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
    with torch.no_grad():  # turn off gradient calculations for speed.
        y = truncated_model(img_tensor)
    yc, xc = np.unravel_index(spatial_index, y.shape[-2:])
    original_response = y[:, unit_i, yc, xc].cpu().detach().numpy()

    occluder_i = 0
    num_stim = len(occluder_params)
    discrepancy_map = np.zeros((rf_size, rf_size))

    while (occluder_i < num_stim):
        # sys.stdout.write('\r')
        # sys.stdout.write(f"Presenting {occluder_i}/{num_stim} stimuli...")
        # sys.stdout.flush()

        real_batch_size = min(batch_size, num_stim-occluder_i)
        occluder_batch = torch.zeros((real_batch_size, 3, *image_size)).to(c.DEVICE)

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = occluder_params[occluder_i + i]
            occluder_batch[i] = draw_random_occluder(img_tensor,
                                                     params['top_left'],
                                                     params['bottom_right'])

        # Present the patch of bars to the truncated model.
        y = truncated_model(occluder_batch)
        y = y.cpu().detach().numpy()
        yc, xc = np.unravel_index(spatial_index, y.shape[-2:])
        response_diff = y[:, unit_i, yc, xc] - original_response
        if _debug:
            # Skip some occluders for testing.
            add_discrepancy_maps(response_diff,
                                occluder_params[occluder_i:occluder_i+batch_size:5], discrepancy_map, box, rf_size)
        else:
            add_discrepancy_maps(response_diff,
                             occluder_params[occluder_i:occluder_i+batch_size], discrepancy_map, box, rf_size)
        occluder_i += real_batch_size

    return discrepancy_map
