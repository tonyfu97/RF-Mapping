"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 22, 2022
"""
import os
from heapq import heapify, heappush, heappushpop, nlargest          


import numpy as np
from torch.nn import Module
import torch.nn as nn
from torchvision.transforms import transforms


from .._utils import layer_finder
from .._utils import num_units_in_layers


class LayerOutputInspector:
    """
    A class that "peeks" inside the output tensors of all the layers with the
    specified layer type, one image at a time.
    """
    def __init__(self, model, layer_type=nn.Conv2d):
        self.model = model
        self.layer_type = layer_type
        self.layer_outputs = []
                
    def hook_function(self, ten_in, ten_out):
        self.layer_outputs.append(ten_out)

    def register_forward_hook_to_layers(self):
        # If "model" is a leave node and matches the layer_type, register hook.
        if (len(list(self.model.children())) == 0):
            if (isinstance(self.model, self.layer_type)):
                self.model.register_forward_hook(self.hook_function)
                return

        # Recurse otherwise.
        else:
            for i, sublayer in enumerate(self.model.children()):
                self.register_forward_hook_to_layers()
                
    def inspect(self, image):
        """
        Given an image, returns the output activation volumes of all the layers
        of the type <layer_type>.

        Parameters
        ----------
        image : numpy.array
            Input image, most likely with the dimension: [1, 3, 227, 227].

        Returns
        -------
        layer_outputs : list of torch.Tensors
            Each item is an output activation volume of a target layer.
        """
        _ = self.model(image)
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


def 





def _init_heaps(self):
        for layer_i, num_units in enumerate(self.num_units_in_layers):
            self.activation_heaps[f"Conv{layer_i}"] = []
            for _ in range(num_units):
                self.activation_heaps[f"Conv{layer_i}"].append(MaxHeap(self.N))


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr