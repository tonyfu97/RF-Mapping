"""
Making gradient ascent visualization. Similar to make_zero_intialized_grad_ascent.py, 
but the images are not initialized with zeros. Instead, they are initialized by
the image patches that give them the highest (most positive) responses.

Logs
----
Model Name: alexnet
Optimization Method: SGD (no momentum)
Device: Macbook Pro 16" Late 2021 with M1 Pro
Time: 37 minutes
Space: generates about 674 MB of data.

Tony Fu, Bair Lab, March 2023

"""

# #################################### GUARD ####################################

# confirmation = input("Are you sure you want to run this code? (Y/N)")
# if confirmation.lower() == "y":
#     pass
# else:
#     print("Code execution aborted.")

# ###############################################################################


import os
import sys
import multiprocessing
from typing import Tuple

import torch
import numpy as np
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.append('../../..')
import src.rf_mapping.constants as c
from src.rf_mapping.spatial import SpatialIndexConverter
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.image import normalize_img
from src.rf_mapping.grad_ascent import GradientAscent
from src.rf_mapping.model_utils import ModelInfo


# Please specify some model details here:
MODEL_NAME = "vgg16"

# Specify optimization method
OPTIMIZATION_METHOD = 'SGD'  # options: SGD and Adam
NUM_ITER = 100
LR = 0.1
MOMENTUM = False

# Set the result directory
RESULT_DIR = os.path.join(c.RESULTS_DIR, 'gradient_ascent','mapping', MODEL_NAME)

########################### DON'T TOUCH CODE BELOW ############################

# Load model and related information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = getattr(models, MODEL_NAME)(pretrained=True).to(DEVICE)
MODEL_INFO = ModelInfo()
LAYER_NAMES = MODEL_INFO.get_layer_names(MODEL_NAME)

# Get the image directory
IMG_SIZE = (227, 227)
NUM_TOP_IMAGES = 10

"""
Obviously, I cannot upload the content of IMG_DIR to GitHub because it is too
big. Here is some more information about the images at IMG_DIR so you can
try it yourself:

- Source: a subset ImageNet's testing set
- Number of image: 50000
- Size: 3 x 227 x 227
- Total size: about 62 GB
- Format: .npy (NumPy arrays)
- Note: The RGB values are batch-normalized to be roughly in the range of -1.0 to +1.0.
- Where you can download it: http://wartburg.biostr.washington.edu/loc/course/artiphys/data/i50k.html
"""

# Get the spatial indicies (for more info, see below)
def get_max_min_indicies(layer_name):
    spatial_index_path = os.path.join(c.RESULTS_DIR, 'ground_truth', 'top_n',
                                      MODEL_NAME, f"{layer_name}.npy")
    return np.load(spatial_index_path).astype(int)
"""
I feel the need to explain what these spatial indices are all about. This was an idea that was developed by Dr. Wyeth Bair and his PhD student, Dr. Dean Pospisil. They presented the 50,000 images to the neural networks. As a reminder, the convolution (or more accurately, cross-correlation) operation slides the unit's kernel along the two spatial dimensions of the input. The first thing they did was to find the spatial locations that produced the maximum (most positive) responses. They repeated this process for all 50,000 images, and then ranked the resulting max locations to find out which image patches gave the strongest responses.

`MAX_MIN_INDICES` contains the results of this ranking for a particular convolutional layer of a model. The array has dimensions [num_units, 100, 4].

`num_units` represents the number of unique kernels in the convolutional layer. For example, Conv1 of AlexNet has 64 unique kernels. The second dimension `k` is 100 because the array stores the top and bottom 100 image patches. The last dimension has a size of 4 because it contains:
(1) `max_img_idx`: the index (ranging from 0 to 49,999) of the k-th most positive response image.
(2) `max_spatial_idx`: the spatial index of the kernel (not the pixel). The kernel is first slided along the x-axis, then y-axis. For instance, a spatial index of 0 corresponds to (0,0), and 1 to (0, 1). We can convert from 1D indexing to 2D using np.unravel_index(spatial_index, (output_height, output_width)).
(3) `min_img_idx`: same as `max_img_idx`, but for the k-th most negative response image.
(4) `min_spatial_idx`: same as `max_spatial_idx`, but for the k-th most negative response image patch.
"""

# Initiate helper objects. This object converts the spatial index from the
# output layer to that of the input layer (i.e., pixel coordinates).
converter = SpatialIndexConverter(MODEL, IMG_SIZE)

##################### Define a few small helper functions #####################

def clip(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def pad_box(box: Tuple[int, int, int, int], padding: int):
    """Makes sure box does not go beyond the image after padding."""
    y_min, x_min, y_max, x_max = box
    new_y_min = clip(y_min-padding, 0, IMG_SIZE[0])
    new_x_min = clip(x_min-padding, 0, IMG_SIZE[1])
    new_y_max = clip(y_max+padding, 0, IMG_SIZE[0])
    new_x_max = clip(x_max+padding, 0, IMG_SIZE[1])
    return new_y_min, new_x_min, new_y_max, new_x_max

def process_tensor(img_tensor: torch.Tensor, normalize=True) -> np.ndarray:
    """
    Converts a tensor to a Numpy array and normalize it to [0, 1].

    Args:
        img_tensor: A tensor of shape (C, H, W).
        normalize: A flag that decides whether the output should be normalized or not.

    Returns:
        A Numpy array of shape (H, W, C).

    """
    img_numpy = img_tensor.detach().cpu().numpy()
    img_numpy = np.squeeze(img_numpy)
    img_numpy = np.transpose(img_numpy, (1, 2, 0))
    
    if normalize:
        # Normalizes pixel values to [0, 1.0]
        img_range = img_numpy.max() - img_numpy.min()
        if not np.isclose(img_range, 0, rtol=0, atol=1e-5):
            img_numpy = (img_numpy - img_numpy.min()) / img_range
    return img_numpy


def one_sided_zero_pad(patch: np.ndarray, desired_size: int, box: Tuple[int, int, int, int]):
    """
    Return original patch if it is the right size. Assumes that the patch
    given is always smaller or equal to the desired size. The box tells us
    the spatial location of the patch on the image.
    """
    if len(patch.shape) != 3 or patch.shape[0] != 3:
        raise ValueError(f"patch must be have shape (3, height, width), but got {patch.shape}")
    if patch.shape[1] == desired_size and patch.shape[2] == desired_size:
        return patch

    vx_min, hx_min, vx_max, hx_max = box
    touching_top_edge = (vx_min <= 0)
    touching_left_edge = (hx_min <= 0)

    padded_patch = np.zeros((3, desired_size, desired_size))
    _, patch_h, patch_w = patch.shape

    if touching_top_edge and touching_top_edge:
        padded_patch[:, -patch_h:, -patch_w:] = patch  # fill from bottom right
    elif touching_top_edge:
        padded_patch[:, -patch_h:, :patch_w] = patch  # fill from bottom left
    elif touching_left_edge:
        padded_patch[:, :patch_h, -patch_w:] = patch  # fill from top right
    else:
        padded_patch[:, :patch_h, :patch_w] = patch  # fill from top left

    return padded_patch

###############################################################################

def create_visualizations_for_layer(layer_name):
    # Determine layer-specific information
    num_units = MODEL_INFO.get_num_units(MODEL_NAME, layer_name)
    layer_index = MODEL_INFO.get_layer_index(MODEL_NAME, layer_name)
    xn = MODEL_INFO.get_xn(MODEL_NAME, layer_name)
    rf_size = MODEL_INFO.get_rf_size(MODEL_NAME, layer_name)
    padding = (xn - rf_size) // 2
    
    # Use the truncated model to save time
    truncated_model = get_truncated_model(MODEL, layer_index)

    # Define the output directory, create it if necessary
    layer_dir = os.path.join(RESULT_DIR, layer_name)
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)

    # Find the top- and bottom-100 image patches ranking of the layer
    max_min_indicies = get_max_min_indicies(layer_name)
    
    # We will also store the results in a numpy array
    result_array = np.zeros((num_units, xn, xn, 3))
    
    for unit_index in tqdm(range(num_units)):
        for top_i in range(NUM_TOP_IMAGES):
            # Get top and bottom image indices and patch spatial indices
            max_n_img_index   = max_min_indicies[unit_index, top_i, 0]
            max_n_patch_index = max_min_indicies[unit_index, top_i, 1]

            # Convert from output spatial index to pixel coordinate
            box = converter.convert(max_n_patch_index, layer_index, 0, is_forward=False)
            
            # Prevent indexing out of range
            y_min, x_min, y_max, x_max = pad_box(box, padding)
            
            # Load the image
            img_path = os.path.join(c.IMG_DIR, f"{max_n_img_index}.npy")
            img_numpy = np.load(img_path)[:, y_min:y_max+1, x_min:x_max+1]
            
            # Pad it to (3, xn, xn) if necessary
            img_numpy = one_sided_zero_pad(img_numpy, xn, (y_min, x_min, y_max, x_max))
            
            # Convert to tensor
            img = torch.from_numpy(img_numpy).type('torch.FloatTensor').unsqueeze(0)
            img.requires_grad = True
            img.to(DEVICE)
            
            # Computer gradient ascent
            ga = GradientAscent(truncated_model, unit_index, img, lr=LR,
                                optimizer=OPTIMIZATION_METHOD, momentum=MOMENTUM)
            for _ in range(NUM_ITER - 1):
                ga.step()
            result_tensor = ga.step()
            
            # Subtract the original image, then convert to numpy array
            result_npy = process_tensor(result_tensor, normalize=False) - img_numpy.transpose(1, 2, 0)
        
            # Save result to an image
            result_array[unit_index] += result_npy

        plt.imshow(normalize_img(result_array[unit_index]))
        plt.axis('off')
        plt.savefig(os.path.join(layer_dir, f"{unit_index}.png"))
        plt.close()

    np.save(os.path.join(RESULT_DIR, f"{layer_name}.npy"), result_array)

if __name__ == '__main__':
    LAYERS_TO_RUN = [f"conv{i}" for i in range(6,11)]
    with multiprocessing.Pool(processes=len(LAYERS_TO_RUN)) as pool:
        pool.map(create_visualizations_for_layer, LAYERS_TO_RUN)
