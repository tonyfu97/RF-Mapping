"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 25, 2022
"""
import os
import sys
from unittest import result

import numpy as np
import torch
import torch.nn as nn
# from torchvision.models import VGG16_Weights
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../../..')
from src.rf_mapping.image import preprocess_img_to_tensor
from src.rf_mapping.hook import HookFunctionBase, ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.reproducibility import set_seeds
import src.rf_mapping.constants as c

# Please specify some details here:
# set_seeds()
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = "vgg16"
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
num_images = 50000
batch_size = 32
top_n = 100
yn, xn = (227, 227)
this_is_a_test_run = True
is_random = False

# Please double-check the directories:
img_dir = c.IMG_DIR
img_names = [f"{i}.npy" for i in range(num_images)]

if is_random:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n_random')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'ground_truth', 'top_n')

if this_is_a_test_run:
    result_dir = os.path.join(result_dir, 'test')
else:
    result_dir = os.path.join(result_dir, model_name)

###############################################################################

# # Script guard.
# if __name__ == "__main__":
#     user_input = input("This code takes time to run. Are you sure? "\
#                        "Enter 'y' to proceed. Type any other key to stop: ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")


class ConvMaxMinInspector(HookFunctionBase):
    """
    A class that get the maximum activations and indices of all unique
    convolutional kernels, one image at a time.
    """
    def __init__(self, model):
        super().__init__(model, nn.Conv2d)
        self.model.eval()  # MUST config model to evaluation mode!
                           # Otherwise, resnet18 performs 'batch' normalization
                           # with running estimates on single images.
        self.img_max_activations = []
        self.img_max_indices = []
        self.img_min_activations = []
        self.img_min_indices = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        batch_size = ten_out.shape[0]
        num_units = ten_out.shape[1]

        layer_max_activations = np.zeros((batch_size, num_units))
        layer_max_indices = np.zeros((batch_size, num_units))
        layer_min_activations = np.zeros((batch_size, num_units))
        layer_min_indices = np.zeros((batch_size, num_units))
        
        for batch_i in range(batch_size):
            for unit in range(num_units):
                layer_max_activations[batch_i, unit] = ten_out[batch_i,unit,:,:].max().item()
                layer_max_indices[batch_i, unit] = ten_out[batch_i, unit,:,:].argmax().item()
                layer_min_activations[batch_i, unit] = ten_out[batch_i,unit,:,:].min().item()
                layer_min_indices[batch_i, unit] = ten_out[batch_i, unit,:,:].argmin().item()
            
        self.img_max_activations.append(layer_max_activations)
        self.img_max_indices.append(layer_max_indices)
        self.img_min_activations.append(layer_min_activations)
        self.img_min_indices.append(layer_min_indices)

    def inspect(self, images):
        """
        Given an image, returns the output activation volumes of all the layers
        of the type <layer_type>.

        Parameters
        ----------
        images : torch.tensor
            Input images, most likely with the dimension: [num_img, 3, 2xx, 2xx].

        Returns
        -------
        max_activations : list of numpy.arrays
            The maximum response.
        """
        with torch.no_grad():  # turn off gradient calculations for speed.
            _ = self.model(images)
        
        # Make copies of the list attributes and set them to empty lists before
        # returning them. Otherwise, they would use up too much memory.
        copy_max_activations = self.img_max_activations[:]
        copy_max_indices = self.img_max_indices[:]
        copy_min_activations = self.img_min_activations[:]
        copy_min_indices = self.img_min_indices[:]
        
        self.img_max_activations = []
        self.img_max_indices = []
        self.img_min_activations = []
        self.img_min_indices = []
        
        return copy_max_activations, copy_min_activations, copy_max_indices, copy_min_indices


# Initiate helper objects.
inspector = ConvMaxMinInspector(model)

# Get info of conv layers.
unit_counter = ConvUnitCounter(model)
layer_indices, nums_units = unit_counter.count()

# Initialize arrays:
all_activations = []
num_layers = len(layer_indices)
for num_units in nums_units:
    all_activations.append(np.zeros((num_images, num_units, 6), dtype=int))


print("Recording responses...")
img_i = 0
while (img_i < num_images):
    real_batch_size = min(num_images - img_i, batch_size)
    sys.stdout.write('\r')
    sys.stdout.write(f"Presenting image no.{img_i}")
    sys.stdout.flush()
    if this_is_a_test_run and img_i > 1000:
        break

    # Prepare image tensor
    img_tensor = torch.zeros((real_batch_size, 3, yn, xn)).to(c.DEVICE)
    for i in range(real_batch_size):
        img_path = os.path.join(img_dir, f"{img_i + i}.npy")
        img = np.load(img_path)
        img_t = preprocess_img_to_tensor(img)
        img_tensor[i] = img_t[0]  # [0] to remove first dimension.
    
    # Present image tensor
    img_max_activations, img_min_activations, img_max_indices, img_min_indices =\
                                            inspector.inspect(img_tensor)
    
    for layer_i in range(num_layers):
        num_units = img_max_activations[layer_i].shape[1]

        layer_max_activations = img_max_activations[layer_i]
        layer_max_indices = img_max_indices[layer_i]
        layer_min_activations = img_min_activations[layer_i]
        layer_min_indices = img_min_indices[layer_i]
        
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 0] = np.tile(np.arange(img_i, img_i+real_batch_size), (num_units, 1)).T
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 1] = np.tile(np.arange(num_units), (real_batch_size, 1))
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 2] = layer_max_activations * 100000
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 3] = layer_max_indices
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 4] = layer_min_activations * 100000
        all_activations[layer_i][img_i:img_i+real_batch_size, :, 5] = layer_min_indices

    img_i += real_batch_size

print("Sorting responses...")
all_sorted_top_n_img_indices = []
all_sorted_responses = []
for layer_i in tqdm(range(num_layers)):
    num_units =  all_activations[layer_i].shape[1]
    top_n_img_idx = np.zeros((num_units, top_n, 4), dtype=int)
    sorted_responses = np.zeros((num_units, top_n, 2))
    for unit_i in range(num_units):
        # Top N patches:
        sorted_img_index = all_activations[layer_i][:,unit_i,2].argsort()
        sorted_img_index = np.flip(sorted_img_index)  # Make it descending
        top_n_img_idx[unit_i, :, 0] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 1] = all_activations[layer_i][:,unit_i,3][sorted_img_index][:top_n]
        sorted_responses[unit_i, :, 0] = all_activations[layer_i][:,unit_i,2].sort() / 100000
        
        # Bottom N patches:
        sorted_img_index = all_activations[layer_i][:,unit_i,4].argsort()
        top_n_img_idx[unit_i, :, 2] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 3] = all_activations[layer_i][:,unit_i,5][sorted_img_index][:top_n]
        sorted_responses[unit_i, :, 1] = all_activations[layer_i][:,unit_i,4].sort() / 100000

    all_sorted_top_n_img_indices.append(top_n_img_idx)
    all_sorted_responses.append(sorted_responses)


print("Saving responses...")
delete_all_npy_files()  # TODO: DELETE LATER, already has os.path.exist() below
for layer_i in tqdm(range(num_layers)):
    result_idx_path = os.path.join(result_dir, f"conv{layer_i+1}.npy")
    if os.path.exists(result_idx_path):
        os.remove(result_idx_path)
    print(result_idx_path)
    np.save(result_idx_path, all_sorted_top_n_img_indices[layer_i])

    result_response_path = os.path.join(result_dir, f"conv{layer_i+1}_responses.npy")
    if os.path.exists(result_response_path):
        os.remove(result_response_path)
    print(result_response_path)
    np.save(result_response_path, all_sorted_top_n_img_indices[layer_i])


num_bins = 30
pdf_path = os.path.join(result_dir, f"{model_name}_summary.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(num_layers*5, 10))
    plt.suptitle(f"Statistics of 50,000 ImageNet Images: Activations", fontsize=26)

    for layer_i in range(num_layers):
        layer_name = f"conv{layer_i+1}"
        plt.subplot(2, num_layers, layer_i+1)
        plt.hist(img_max_activations[layer_i], bins=num_bins)
        plt.xlabel("max responses", fontsize=14)
        plt.title(layer_name, fontsize=18)
        
        plt.subplot(2, num_layers, layer_i+1+num_layers)
        plt.hist(img_min_activations[layer_i], bins=num_bins)
        plt.xlabel("min responses", fontsize=14)
        plt.title(layer_name, fontsize=18)

    pdf.savefig()
    plt.close()
