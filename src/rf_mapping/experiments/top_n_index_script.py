"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 25, 2022
"""
import os
import sys

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

sys.path.append('..')
from image import preprocess_img_to_tensor
from hook import HookFunctionBase, SpatialIndexConverter, ConvUnitCounter
from files import delete_all_npy_files

device = ('mps' if torch.has_mps else 'cpu')

# Please specify some details here:
model = models.vgg16(pretrained=True).to(device)
model_name = "vgg16"  
num_images = 50000
top_n = 100

# Please double-check the directories:
# img_dir = Path(__file__).parent.parent.parent.joinpath('data/imagenet')
img_dir = "/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images"
img_names = [f"{i}.npy" for i in range(num_images)]
result_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/ground_truth/top_n/{model_name}')

###############################################################################

# Script guard.
if __name__ == "__main__":
    user_input = input("This code takes time to run. Are you sure? "\
                       "Enter 'y' to proceed. Type any other key to stop: ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")


class ConvMaxMinInspector(HookFunctionBase):
    """
    A class that get the maximum activations and indicies of all unique
    convolutional kernels, one image at a time.
    """
    def __init__(self, model):
        super().__init__(model, nn.Conv2d)
        self.img_max_activations = []
        self.img_max_indicies = []
        self.img_min_activations = []
        self.img_min_indicies = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        layer_max_activations = np.zeros(ten_out.shape[1])
        layer_max_indicies = np.zeros(ten_out.shape[1])
        layer_min_activations = np.zeros(ten_out.shape[1])
        layer_min_indicies = np.zeros(ten_out.shape[1])
        
        for unit in range(ten_out.shape[1]):
            layer_max_activations[unit] = ten_out[0,unit,:,:].max().item()
            layer_max_indicies[unit] = ten_out[0,unit,:,:].argmax().item()
            layer_min_activations[unit] = ten_out[0,unit,:,:].min().item()
            layer_min_indicies[unit] = ten_out[0,unit,:,:].argmin().item()
            
        self.img_max_activations.append(layer_max_activations)
        self.img_max_indicies.append(layer_max_indicies)
        self.img_min_activations.append(layer_min_activations)
        self.img_min_indicies.append(layer_min_indicies)

    def inspect(self, image):
        """
        Given an image, returns the output activation volumes of all the layers
        of the type <layer_type>.

        Parameters
        ----------
        image : numpy.array or torch.tensor
            Input image, most likely with the dimension: [3, 2xx, 2xx].

        Returns
        -------
        max_activations : list of numpy.arrays
            The ma
        """
        if not isinstance(image, torch.Tensor):
            image = preprocess_img_to_tensor(image)
        _ = self.model(image)
        
        # Make copies of the list attributes and set them to empty lists before
        # returning them. Otherwise, they would use up too much memory.
        copy_max_activations = self.img_max_activations[:]
        copy_max_indicies = self.img_max_indicies[:]
        copy_min_activations = self.img_min_activations[:]
        copy_min_indicies = self.img_min_indicies[:]
        
        self.img_max_activations = []
        self.img_max_indicies = []
        self.img_min_activations = []
        self.img_min_indicies = []
        
        return copy_max_activations, copy_min_activations, copy_max_indicies, copy_min_indicies


# Initiate helper objects.
converter = SpatialIndexConverter(model, (227, 227))
inspector = ConvMaxMinInspector(model)

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indicies, nums_units = unit_counter.count()

# Initialize arrays:
all_activations = []
num_layers = len(layer_indicies)
for num_units in nums_units:
    all_activations.append(np.zeros((num_images, num_units, 6), dtype=int))


print("Recording responses...")
for img_i, img_name in enumerate(tqdm(img_names)):
    img_path = os.path.join(img_dir, img_name)
    img = np.load(img_path)
    img_max_activations, img_min_activations, img_max_indicies, img_min_indicies =\
                                                inspector.inspect(img)
    
    for layer_i in range(num_layers):
        num_units = len(img_max_activations[layer_i])
        
        layer_max_activations = img_max_activations[layer_i]
        layer_max_indicies = img_max_indicies[layer_i]
        layer_min_activations = img_min_activations[layer_i]
        layer_min_indicies = img_min_indicies[layer_i]
        
        all_activations[layer_i][img_i, :, 0] = img_i
        all_activations[layer_i][img_i, :, 1] = np.arange(num_units)
        all_activations[layer_i][img_i, :, 2] = layer_max_activations * 100000
        all_activations[layer_i][img_i, :, 3] = layer_max_indicies
        all_activations[layer_i][img_i, :, 4] = layer_min_activations * 100000
        all_activations[layer_i][img_i, :, 5] = layer_min_indicies


print("Sorting responses...")
sorted_activations = []
for layer_i in tqdm(range(num_layers)):
    num_units =  all_activations[layer_i].shape[1]
    top_n_img_idx = np.zeros((num_units, top_n, 4), dtype=int)
    for unit_i in range(num_units):
        # Top N patches:
        sorted_img_index = all_activations[layer_i][:,unit_i,2].argsort()
        sorted_img_index = np.flip(sorted_img_index)  # Make it descending
        top_n_img_idx[unit_i, :, 0] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 1] = all_activations[layer_i][:,unit_i,3][sorted_img_index][:top_n]
        
        # Bottom N patches:
        sorted_img_index = all_activations[layer_i][:,unit_i,4].argsort()
        top_n_img_idx[unit_i, :, 2] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 3] = all_activations[layer_i][:,unit_i,5][sorted_img_index][:top_n]

    sorted_activations.append(top_n_img_idx)


print("Saving responses...")
delete_all_npy_files(result_dir)
for layer_i in tqdm(range(num_layers)):
    result_path = os.path.join(result_dir, f"conv{layer_i+1}.npy")
    print(result_path)
    np.save(result_path, sorted_activations[layer_i])
