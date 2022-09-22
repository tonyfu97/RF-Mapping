"""
Code to compute the Color Rotation Indices (CRI) of the units.

Tony Fu, August 19, 2022
"""
import os
import sys
import random
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
# from torchvision.models import VGG16_Weights
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../..')
from src.rf_mapping.image import preprocess_img_to_tensor
from src.rf_mapping.hook import HookFunctionBase, ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
import src.rf_mapping.constants as c

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
num_images = 1000
batch_size = 50
this_is_a_test_run = True

# Please double-check the directories:
img_dir = c.IMG_DIR
result_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'cri', model_name)

###############################################################################

# Script guard.
# if __name__ == "__main__":
#     user_input = input("This code takes time to run. Are you sure? "\
#                        "Enter 'y' to proceed. Type any other key to stop: ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")


class ColorRotationInspector(HookFunctionBase):
    """
    A class that get the maximum activations and indices of all unique
    convolutional kernels, one image at a time.
    """
    def __init__(self, model):
        super().__init__(model, nn.Conv2d)
        self.model.eval()  # MUST config model to evaluation mode!
                           # Otherwise, resnet18 performs 'batch' normalization
                           # with running estimates on single images.
        self.center_responses = []
        self.rotation_stds = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        mid_y = (ten_out.shape[2] - 1) // 2
        mid_x = (ten_out.shape[3] - 1) // 2
        self.center_responses.append(ten_out[:, :, mid_y, mid_x].detach().cpu().numpy())
    
    def _get_all_image_permututations(self, images):
        all_permutations = []
        for (ri, gi, bi) in permutations([0, 1, 2]):
            perm_images = torch.empty(images.shape).to(c.DEVICE)
            perm_images[:,ri,...] = images[:,0,...]
            perm_images[:,gi,...] = images[:,1,...]
            perm_images[:,bi,...] = images[:,2,...]
            all_permutations.append(perm_images)
        return all_permutations

    def inspect(self, images):
        """
        Given an image, returns the output activation volumes of all the layers
        of the type <layer_type>.

        Parameters
        ----------
        images : torch.tensor
            Input images with the dimension: [num_img, 3, 2xx, 2xx].

        Returns
        -------
        max_activations : list of numpy.arrays
            The maximum response.
        """
        all_permutations = self._get_all_image_permututations(images)
        six_all_center_responses = []
        for perm_i, permutation in enumerate(all_permutations):
            self.center_responses = []  # clear previous responses

            with torch.no_grad():  # turn off gradient calculations for speed.
                _ = self.model(permutation)
            
            if perm_i == 0:  # If this is the first forward pass (with original RGB)
                original_center_responses = self.center_responses.copy()
                for conv_i, layer_center_responses in enumerate(self.center_responses):
                    # Initialize the center responses array for all permutations
                    six_all_center_responses.append(np.zeros((6, *layer_center_responses.shape)))
            
            # Record the center responses of all layers and all permutation
            for conv_i, layer_center_responses in enumerate(self.center_responses):
                six_all_center_responses[conv_i][perm_i] = layer_center_responses
        
        # Calculate the image-wise standard deviation for all 6 rotations
        image_wise_standard_devations = []
        for conv_i, layer_center_responses in enumerate(six_all_center_responses):
            image_wise_standard_devations.append(np.std(six_all_center_responses[conv_i], axis=0))
        
        return original_center_responses, image_wise_standard_devations


# Initiate helper objects.
inspector = ColorRotationInspector(model)

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()

# Initialize arrays:
all_original_center_responses = []
all_image_wise_standard_devations = []
num_layers = len(layer_indices)
for num_units in nums_units:
    all_original_center_responses.append(np.zeros((num_images, num_units)))
    all_image_wise_standard_devations.append((np.zeros((num_images, num_units))))

# Randomly sample <num_images> number of images
random.seed(123)  # For reproducibility
img_list = random.choices(range(50000), k=num_images)

print("Recording responses...")
img_i = 0
while (img_i < num_images):
    real_batch_size = min(num_images - img_i, batch_size)
    sys.stdout.write('\r')
    sys.stdout.write(f"Presenting {img_i} images")
    sys.stdout.flush()
    if this_is_a_test_run and img_i > 1000:
        break

    # Prepare image tensor
    img_tensor = torch.zeros((real_batch_size, 3, 227, 227)).to(c.DEVICE)
    for i in range(real_batch_size):
        img_path = os.path.join(img_dir, f"{img_list[img_i + i]}.npy")
        img = np.load(img_path)
        img_t = preprocess_img_to_tensor(img)
        img_tensor[i] = img_t[0]  # [0] to remove first dimension.
    
    # Present image tensor
    original_center_responses, image_wise_standard_devations =\
                                            inspector.inspect(img_tensor)

    # Put the center responses and standard deviations of all images to one
    # array, repeat for all layers.
    for conv_i in range(num_layers):
        all_original_center_responses[conv_i][img_i:img_i+real_batch_size] = original_center_responses[conv_i]
        all_image_wise_standard_devations[conv_i][img_i:img_i+real_batch_size] = image_wise_standard_devations[conv_i]

    img_i += real_batch_size


print("Calculate Color Rotation Index (CRI)...")
cri_txt_path = os.path.join(result_dir, f"cri.txt")
if os.path.exists(cri_txt_path):
    os.remove(cri_txt_path)
for conv_i in range(len(nums_units)):
    numerators = np.mean(all_image_wise_standard_devations[conv_i], axis=0)
    denominators = np.std(all_original_center_responses[conv_i], axis=0)
    cri_list = numerators/denominators
    
    for unit_i, cri in enumerate(cri_list):
        with open(cri_txt_path, 'a') as f:
            f.write(f"conv{conv_i+1} {unit_i} {cri:.4f}\n")
