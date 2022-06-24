"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 22, 2022
"""
import os
import math
from heapq import heapify, heappush, heappushpop, nlargest    
from collections import OrderedDict   


import numpy as np
# from numpy import unravel_index
import torch
import torch.nn as nn
from torchvision import models


class HookFunctionBase:
    """
    A base class that register a hook function to all specified layer types
    in a given model. The child class must implement hook_function(). The child
    class must also call self.register_forward_hook_to_layers() by itself.
    """
    def __init__(self, model, layer_types):
        self.model = model
        self.layer_types = layer_types
        
    def hook_function(self, module, ten_in, ten_out):
        raise NotImplementedError

    def register_forward_hook_to_layers(self, layer):
        # If "model" is a leave node and matches the layer_type, register hook.
        if (len(list(layer.children())) == 0):
            if (isinstance(layer, self.layer_types)):
                layer.register_forward_hook(self.hook_function)

        # ...recurse otherwise.
        else:
            for i, sublayer in enumerate(layer.children()):
                self.register_forward_hook_to_layers(sublayer)


class LayerOutputInspector(HookFunctionBase):
    """
    A class that "peeks" inside the outputs of all the layers with the
    specified layer type, one image at a time.
    """
    def __init__(self, model, layer_types=(nn.Conv2d)):
        super().__init__(model, layer_types)
        self.layer_outputs = []
        self.register_forward_hook_to_layers(self.model)
        
    def hook_function(self, module, ten_in, ten_out):
        self.layer_outputs.append(ten_out.clone().detach().numpy())
                
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


class SizeInspector(HookFunctionBase):
    """
    A class that computes input and output sizes of the specified layers.
    """
    def __init__(self, model, image_shape, layer_types):
        super().__init__(model, layer_types)
        self.model = model
        self.image_shape = image_shape
        self.layers = []
        self.input_sizes = []
        self.output_sizes = []
        self.register_forward_hook_to_layers(self.model)
        self.model(torch.zeros((1,3,*image_shape)))
        
    def hook_function(self, module, ten_in, ten_out):
        if (isinstance(module, self.layer_types)):
            self.layers.append(module)
            self.input_sizes.append(ten_in[0].shape)
            self.output_sizes.append(ten_out.shape)
        
    def print_summary(self):
        for i, layer in enumerate(self.layers):
            print("---------------------------------------------------------")
            print(f"  layer no.{i}: {layer}")
            print(f"  input size: ({self.input_sizes[i][2]}, {self.input_sizes[i][3]})")
            print(f" output size: ({self.output_sizes[i][2]}, {self.output_sizes[i][3]})")


if __name__ == '__main__':
    model = models.alexnet()
    inspector = SizeInspector(model, (227, 227), (nn.Conv2d, nn.MaxPool2d))
    inspector.print_summary()


class SpatialIndexConvertor(SizeInspector):
    """
    A class containing the model- and image-shape-specific transformations
    of the spatial indicies across different layers. Useful for receptive
    field mapping and other tasks that involve the mappings of spatial
    locations onto an shallower or deeper layer.
    
    Given a spatial location, the projection methods will return a "box"
    in (vx_min, hx_min, vx_max, hx_max) format. Note that the returned point(s)
    here are cooridinates with respect to the destination layer.
    """
    def __init__(self, model, image_shape, layer_types):
        """
        Constructs a SpatialIndexConversion object.

        Parameters
        ----------
        model : torchvision.models
            The neural network.
        image_shape : tuple of ints
            (vertical_dimension, horizontal_dimension) in pixels.
        """
        super().__init__(model, image_shape, layer_types)
        self.rf_sizes = []
        self.dont_need_conversion = (nn.Sequential,
                                    nn.ModuleList,
                                    nn.Sigmoid,
                                    nn.ReLU,
                                    nn.Tanh,
                                    nn.Softmax2d,
                                    nn.BatchNorm2d,
                                    nn.Dropout2d,)
        self.need_convsersion = (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)

    def clip(x, x_min, x_max):
        """Limits x to be x_min <= x <= x_max."""
        x = min(x_max, x)
        x = max(x_min, x)
        return x
        
    def _one_forward_projection(self, layer_index, vx_min, hx_min, vx_max, hx_max):
        """
        Maps the box bounded by (vx_min, hx_min, vx_max, hx_max) onto a deeper
        layer with the specified index <layer_index>. The layer_index does not
        include container layers like torch.nn.Sequential.
        """
        layer = self.layers[layer_index]
        output_size = self.output_sizes[layer_index]
        
        if (isinstance(layer, self.dont_need_conversion)):
            return vx_min, hx_min, vx_max, hx_max
        
        if (isinstance(layer, self.need_convsersion)):
            def transform(x_min, x_max, stride, kernel_size, padding, max_size):  
                x_min = math.floor((x_min - kernel_size)/stride + 1)
                x_min = self.clip(x_min + padding, 0, max_size)
                x_max = math.floor(x_max/stride)
                x_max = self.clip(x_max + padding, 0, max_size)
                return x_min, x_max

            vx_min, vx_max = transform(vx_min, vx_max,
                                       layer.stride[0], layer.kernel_size[0],
                                       layer.padding[0], output_size[0])
            hx_min, hx_max = transform(hx_min, hx_max,
                                       layer.stride[1], layer.kernel_size[1],
                                       layer.padding[1], output_size[1])
            return vx_min, hx_min, vx_max, hx_max

        print(f"{type(layer)} is currently not supported.")
        raise ValueError

    def _one_backward_projection(self, layer_index, vx_min, hx_min, vx_max, hx_max):
        """
        Maps the box bounded by (vx_min, hx_min, vx_max, hx_max) onto a
        shallower layer with the specified index <layer_index>. The layer_index
        does not include container layers like torch.nn.Sequential.
        """
        layer = self.layers[layer_index]
        input_size = self.input_sizes[layer_index]

        if (isinstance(layer, nn.Conv2d)):
            if (layer.dilation != 1):
                print("Dilated convolution is currently not supported.")
                raise ValueError

            def transform(x, stride, kernel_size, padding):
                return (x*stride) + (kernel_size-1)/2 - padding

            vx = transform(vx, layer.stride[0], layer.kernel_size[0], layer.padding[0])
            hx = transform(hx, layer.stride[1], layer.kernel_size[1], layer.padding[1])
            return vx, hx
        
        if (isinstance(layer, nn.MaxPool2d)):
            def transform(x, stride, kernel_size, padding):
                return (x*stride) + (kernel_size-1)/2 - padding
            
            vx = transform(vx, layer.stride[0], layer.kernel_size[0], layer.padding[0])
            hx = transform(hx, layer.stride[1], layer.kernel_size[1], layer.padding[1])
            return vx, hx
        
        return vx, hx
    
    def _process_index(self, index):
        """
        Make sure that the index is a tuple of two indicies. Unravel from 1D
        to 2D indexing if necessary.
        
        Returns
        -------
        index : tuple of ints
            The spatial coordinates of the point of interest in 
            (vertical index, horizontal index) format.
        """
        if index.isnumeric():
            return np.unravel_index(index, self.image_shape)

        if (len(index)==2):
            return index
    
    def forward_projection(self, index, start_layer, end_layer):
        vx, hx = self._process_index(index)
        
        
    def backward_projection(self, index, start_layer, end_layer=0):
        vx, hx = self._process_index(index)