"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 22, 2022
"""
import os
from heapq import heapify, heappush, heappushpop, nlargest
from pprint import PrettyPrinter          


import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import models


class LayerOutputInspector:
    """
    A class that "peeks" inside the outputs of all the layers with the
    specified layer type, one image at a time.
    """
    def __init__(self, model, layer_type=nn.Conv2d):
        self.model = model
        self.layer_type = layer_type
        self.layer_outputs = []
        self.register_forward_hook_to_layers(self.model)
        
    def hook_function(self, module, ten_in, ten_out):
        self.layer_outputs.append(ten_out.clone().detach().numpy())

    def register_forward_hook_to_layers(self, layer):      
        # If "model" is a leave node and matches the layer_type, register hook.
        if (len(list(layer.children())) == 0):
            if (isinstance(layer, self.layer_type)):
                layer.register_forward_hook(self.hook_function)

        # Recurse otherwise.
        else:
            for i, sublayer in enumerate(layer.children()):
                self.register_forward_hook_to_layers(sublayer)
                
    def inspect(self, image):
        """
        Given an image, returns the output activation volumes of all the layers
        of the type <layer_type>.

        Parameters
        ----------
        image : numpy.array
            Input image, most likely with the dimension: [3, 2xx, 2xx].

        Returns
        -------
        layer_outputs : list of numpy.arrays
            Each item is an output activation volume of a target layer.
        """
        # Image preprocessing
        norm_image = image - image.min()
        norm_image = norm_image/norm_image.max()
        norm_image = np.expand_dims(norm_image, axis=0)
        image_tensor = torch.from_numpy(norm_image).type('torch.FloatTensor')
        
        # Forward pass
        _ = self.model(image_tensor)
        
        return self.layer_outputs


class MaxHeap():
    """
    A priority queue with fixed size.
    Credit: @CyanoKobalamyne on stackoverflow.
    """
    def __init__(self, N):
        self.h = []
        self.length = N
        heapify(self.h)
        
    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)
            
    def getTop(self):
        return nlargest(self.length, self.h)


def top_bottom_N_image_patches(model, layer_type, image_dir, image_names):
    model.eval()
    inspector = LayerOutputInspector(model, layer_type)
    image = np.load(f"{image_dir}/{image_names[0]}")
    layer_outputs = inspector.inspect(image)
    for layer_output in layer_outputs: 
        print(layer_output.max())
    

if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    repo_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    image_dir = f"{repo_dir}/data/imagenet"
    image_names = ["0.npy"]
    top_bottom_N_image_patches(model, nn.Conv2d, image_dir, image_names)
