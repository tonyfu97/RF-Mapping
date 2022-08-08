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

Tony Fu, Aug 5, 2022
"""
import os
import sys

import numpy as np
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../..')
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.image import preprocess_img_for_plot, make_box
from src.rf_mapping.spatial import SpatialIndexConverter
from src.rf_mapping.guided_backprop import GuidedBackprop
import src.rf_mapping.constants as c

# Please specify some details here:
# model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(c.DEVICE)
# # model_name = "alexnet"
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
model = models.resnet18(pretrained=True).to(c.DEVICE)
model_name = "resnet18"
top_n = 5
image_size = (227, 227)

# Please double-check the directories:
img_dir = c.IMG_DIR
index_dir = c.REPO_DIR + f'/results/occlude/mapping/{model_name}'
result_dir = index_dir

###############################################################################

# Script guard
if __name__ == "__main__":
    print("Look for a prompt.")
    user_input = input("This code may take hours to run. Are you sure? [y/n] ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

# Initiate helper objects.
converter = SpatialIndexConverter(model, image_size)
conv_counter = ConvUnitCounter(model)

# Get info of conv layers.
layer_indices, nums_units = conv_counter.count()


#######################################.#######################################
#                                                                             #
#                                DRAW_OCCLUDER                                #
#                                                                             #
###############################################################################
def draw_occluder(img, top_left, bottom_right):
    """
    Returns an occluded version of img without modifying the original img.
    Note: the y-axis points downward.

    Parameters
    ----------
    img : numpy.ndarray
        Image with pixel values comparable to [0, 1] and with the color
        channel as the first dimension.
    top_left : (int, int)
        Spatial index of the top left corner (inclusive) of the occluder.
    bottom_right : (int, int))
        Spatial index of the bottom right corner (inclusive) of the occluder.

    Returns
    -------
    occluded_img : numpy.ndarray
        img but with random occluder at the specified location.
    """
    occluded_img = img.copy()
    occluder_size = (bottom_right[0] - top_left[0] + 1,
                     bottom_right[1] - top_left[1] + 1)
    occluder = np.random.rand(3, *occluder_size)
    occluded_img[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = occluder
    
    return occluded_img


# Plot an example occluder.
if __name__ == "__main__":
    img = np.zeros((3, 5, 10))
    occluded_img = draw_occluder(img, (0, 3), (3, 7))
    plt.imshow(np.transpose(occluded_img, (1,2,0)))


#######################################.#######################################
#                                                                             #
#                               OCCLUDER_PARMS                                #
#                                                                             #
###############################################################################
def get_occluder_params(box, rf_size):
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

    for i in np.arange(vx_min, max(vx_max-occluder_size, vx_min+1), occluder_stride):
        for j in np.arange(hx_min, max(hx_max-occluder_size, hx_min+1), occluder_stride):
            occluder_params.append({'top_left' : (i, j), 
                                    'bottom_right' : (i+occluder_size-1, j+occluder_size-1)})
    return occluder_params


# Plot the occulder boxes.
if __name__ == "__main__":
    box = (10, 20, 19, 29)
    occluder_params = get_occluder_params(box, 20)
    print(len(occluder_params))
    plt.ylim([5, 25])
    plt.xlim([15, 35])
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
#                                GET_RESPONSES                                #
#                                                                             #
###############################################################################
def get_discrepancy_maps(img, occluder_params, truncated_model, num_units,
                         batch_size=100, _debug=False):
    """
    Presents the occluded image and returns the discrepency maps (one for
    each unit in the final layer of the truncated model).

    Parameters
    ----------
    img : numpy.ndarray
        
    """
    bar_i = 0
    num_stim = len(splist)
    xn = splist[0]['xn']
    yn = splist[0]['yn']
    center_responses = np.zeros((num_stim, num_units))

    while (bar_i < num_stim):
        if _debug and bar_i > 200:
            break
        print_progress(f"Presenting {bar_i}/{num_stim} stimuli...")
        real_batch_size = min(batch_size, num_stim-bar_i)
        bar_batch = np.zeros((real_batch_size, 3, yn, xn))

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = splist[bar_i + i]
            new_bar = stimfr_bar(params['xn'], params['yn'],
                                 params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'], 
                                params['aa'], params['fgval'], params['bgval'])
            # Replicate new bar to all color channel.
            bar_batch[i, 0] = new_bar
            bar_batch[i, 1] = new_bar
            bar_batch[i, 2] = new_bar

        # Present the patch of bars to the truncated model.
        y = truncated_model(torch.tensor(bar_batch).type('torch.FloatTensor').to(c.DEVICE))
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[bar_i:bar_i+real_batch_size, :] = y[:, :, yc, xc].detach().cpu().numpy()
        bar_i += real_batch_size

    return center_responses

for conv_i, layer_idx in enumerate(layer_indices):
    layer_name = f"conv{conv_i + 1}"
    index_path = os.path.join(index_dir, f"{layer_name}.npy")
    max_min_indices = np.load(index_path).astype(int)
    # with dimension: [units, top_n_img, [max_img_idx, max_idx, min_img_idx, min_idx]]

    num_units = nums_units[conv_i]
    print(f"Making pdf for {layer_name}...")

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
                im_handles.append(ax.imshow(np.zeros((*image_size, 3))))

        for unit_i in tqdm(range(num_units)):
            fig.suptitle(f"{layer_name} unit no.{unit_i}", fontsize=20)
            # Get top and bottom image indices and patch spatial indices
            max_n_img_indices   = max_min_indices[unit_i, :top_n, 0]
            max_n_patch_indices = max_min_indices[unit_i, :top_n, 1]
            min_n_img_indices   = max_min_indices[unit_i, :top_n, 2]
            min_n_patch_indices = max_min_indices[unit_i, :top_n, 3]

            # Top N images and gradient patches:
            for i, (max_img_idx, max_patch_idx) in enumerate(zip(max_n_img_indices,
                                                              max_n_patch_indices)):
                box = converter.convert(max_patch_idx, layer_idx, 0, is_forward=False)
                plot_one_img(im_handles[i], ax_handles[i], max_img_idx, box)
                ax_handles[i].set_title(f"top {i+1} image")

            # Bottom N images and gradient patches:
            for i, (min_img_idx, min_patch_idx) in enumerate(zip(min_n_img_indices,
                                                            min_n_patch_indices)):
                box = converter.convert(min_patch_idx, layer_idx, 0, is_forward=False)
                plot_one_img(im_handles[i+top_n], ax_handles[i+top_n], min_img_idx, box)
                ax_handles[i+top_n].set_title(f"bottom {i+1} image")

            pdf.savefig(fig)
            plt.close()
