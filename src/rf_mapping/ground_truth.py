"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 25, 2022
"""
import os


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


from image import ImageNetDataset, preprocess_img
from hook import HookFunctionBase, SpatialIndexConverter


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
        
        """
        if (not isinstance(image, torch.Tensor)):
            image = preprocess_img(image)
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


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained=True).to(device)  # Remember to change the model name, too!
model_name = "alexnet"  

num_images = 100
# img_dir = Path(__file__).parent.parent.parent.joinpath('data/imagenet')
img_dir = Path(_-)
img_names = [f"{i}.npy" for i in range(num_images)]

imagenet_data = ImageNetDataset(img_dir, img_names)
converter = SpatialIndexConverter(model, (227, 227))
inspector = ConvMaxMinInspector(model)

# Initialize arrays:
all_activations = []
test_img = next(iter(imagenet_data))[0]
img_activations, _, _, _ = inspector.inspect(test_img.to(device))
num_layers = len(img_activations)
for layer_activations in img_activations:
    num_units = len(layer_activations)
    all_activations.append(np.zeros((num_images, num_units, 6), dtype=int))

print("Recording responses...")
for img_i, (img, label) in enumerate(tqdm(imagenet_data)):
    img_max_activations, img_min_activations, img_max_indicies, img_min_indicies =\
                                                inspector.inspect(img.to(device))
    
    for layer_i in range(num_layers):
        num_units = len(img_max_activations[layer_i])
        
        layer_max_activations = img_max_activations[layer_i]
        layer_max_indicies = img_max_indicies[layer_i]
        layer_min_activations = img_min_activations[layer_i]
        layer_min_indicies = img_min_indicies[layer_i]
        
        all_activations[layer_i][img_i, :, 0] = img_i
        all_activations[layer_i][img_i, :, 1] = np.arange(num_units)
        all_activations[layer_i][img_i, :, 2] = layer_max_activations * 10000
        all_activations[layer_i][img_i, :, 3] = layer_max_indicies
        all_activations[layer_i][img_i, :, 4] = layer_min_activations * 10000
        all_activations[layer_i][img_i, :, 5] = layer_min_indicies


print("Sorting responses...")
sorted_activations = []
top_n = 100
for layer_i in tqdm(range(num_layers)):
    num_units =  all_activations[layer_i].shape[1]
    top_n_img_idx = np.zeros((num_units, top_n, 4), dtype=int)
    for unit_i in range(num_units):
        sorted_img_index = all_activations[layer_i][:,unit_i,2].argsort()
        sorted_img_index = np.flip(sorted_img_index)  # Make it descending
        top_n_img_idx[unit_i, :, 0] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 1] = all_activations[layer_i][:,unit_i,3][sorted_img_index][:top_n]
        
        sorted_img_index = all_activations[layer_i][:,unit_i,4].argsort()
        top_n_img_idx[unit_i, :, 2] = all_activations[layer_i][:,unit_i,0][sorted_img_index][:top_n]
        top_n_img_idx[unit_i, :, 3] = all_activations[layer_i][:,unit_i,5][sorted_img_index][:top_n]

    sorted_activations.append(top_n_img_idx)

def delete_all_npy_files(dir):
    for f in os.listdir(dir):
        if f.endswith('.npy'):
            os.remove(os.path.join(dir, f))

print("Saving responses...")
result_dir = Path(__file__).parent.parent.parent.joinpath(f'results/ground_truth/{model_name}')
delete_all_npy_files(result_dir)
for layer_i in tqdm(range(num_layers)):
    result_path = os.path.join(result_dir, f"conv{layer_i+1}.npy")
    print(result_path)
    np.save(result_path, sorted_activations[layer_i])
