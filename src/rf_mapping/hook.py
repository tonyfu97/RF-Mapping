"""
Functions that register hooks for a variety of purposes.

Tony Fu, Jun 22, 2022
"""
import os
import copy


import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from image import preprocess_img_to_tensor


class HookFunctionBase:
    """
    A base class that register a hook function to all specified layer types
    (excluding all container types) in a given model. The child class must
    implement hook_function(). The child class must also call
    self.register_forward_hook_to_layers() by itself.
    """
    def __init__(self, model, layer_types):
        """
        Constructs a HookFunctionBase object.

        Parameters
        ----------
        model : torchvision.models
            The neural network.
        layer_types : tuple of torch.nn.Modules
            A tuple of the layer types you would like to register the forward
            hook to. For example, layer_types = (nn.Conv2d, nn.ReLU) means
            that all the Conv2d and ReLU layers will be registered with the
            forward hook.
        """
        self.model = copy.deepcopy(model)
        self.layer_types = layer_types

    def hook_function(self, module, ten_in, ten_out):
        raise NotImplementedError("Child class of HookFunctionBase must "
                                  "implement hookfunction(self, module, ten_in, ten_out)")

    def register_forward_hook_to_layers(self, layer):
        # If "model" is a leave node and matches the layer_type, register hook.
        if (len(list(layer.children())) == 0):
            if (isinstance(layer, self.layer_types)):
                layer.register_forward_hook(self.hook_function)

        # Otherwise (i.e.,the layer is a container type layer), recurse.
        else:
            for i, sublayer in enumerate(layer.children()):
                self.register_forward_hook_to_layers(sublayer)


class LayerOutputInspector(HookFunctionBase):
    """
    A class that peeks inside the outputs of all the layers with the
    specified layer type, one image at a time.
    """
    def __init__(self, model, layer_types=(nn.Conv2d)):
        super().__init__(model, layer_types)
        self.layer_outputs = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        self.layer_outputs.append(ten_out.clone().detach())

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
        layer_outputs : list of torch.tensors
            Each item is an output activation volume of a target layer.
        """
        if (not isinstance(image, torch.Tensor)):
            image = preprocess_img_to_tensor(image)
        _ = self.model(image)
        return self.layer_outputs
    

class ConvMaxInspector(HookFunctionBase):
    """
    A class that get the maximum activations and indicies of all unique
    convolutional kernels, one image at a time.
    """
    def __init__(self, model):
        super().__init__(model, nn.Conv2d)
        self.all_max_activations = []
        self.all_max_indicies = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        layer_max_activations = []
        layer_max_indicies = []
        
        for unit in range(ten_out.shape[1]):
            layer_max_activations.append(ten_out[0,unit,:,:].max().item())
            layer_max_indicies.append(ten_out[0,unit,:,:].max().item())
            
        self.all_max_activations.append(layer_max_activations)
        self.all_max_indicies.append(layer_max_indicies)

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
        layer_outputs : list of torch.tensors
            Each item is an output activation volume of a target layer.
        """
        if (not isinstance(image, torch.Tensor)):
            image = preprocess_img_to_tensor(image)
        _ = self.model(image)
        
        copy_activations = self.all_max_activations[:]
        copy_indicies = self.all_max_indicies[:]
        
        self.all_max_activations = []
        self.all_max_indicies = []
        
        return copy_activations, copy_indicies


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
    A class that computes the input and output sizes of all layers. This
    class determines the indexing convention of the layers. The indexing
    follows the flow of data through the model and excludes all container-type
    layers. For example, the indexing of torchvision.models.alexnet() is:

          no. | layer name
        ------+-----------
           0  |   Conv1
           1  |   ReLU1
           2  |  MaxPool1
           3  |   Conv2
           4  |   ReLU2
             ...
          19  |   ReLU7
          20  |  Linear3

    To get the indexing information for any arbitrary model, use the syntax:
        inspector = SizeInspector(model, image_size)
        inspector.print_summary()
    """
    def __init__(self, model, image_shape):
        super().__init__(model, layer_types=(torch.nn.Module))
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
            self.input_sizes.append(ten_in[0].shape[1:])
            self.output_sizes.append(ten_out.shape[1:])

    def print_summary(self):
        for i, layer in enumerate(self.layers):
            print("---------------------------------------------------------")
            print(f"  layer no.{i}: {layer}")
            try:
                print(f"  input size: ({self.input_sizes[i][0]}, "\
                      f"{self.input_sizes[i][1]}, {self.input_sizes[i][2]})")
                print(f" output size: ({self.output_sizes[i][0]}, "
                      f"{self.output_sizes[i][1]}, {self.output_sizes[i][2]})")
            except:
                print(" This layer is not 2D.")


if __name__ == '__main__':
    model = models.alexnet()
    inspector = SizeInspector(model, (227, 227))
    inspector.print_summary()


def rf_sizes(model, image_shape):
    """
    Find the receptive field (RF) sizes of all layers (excluding containers).
    The sizes here are in pixels with respect to the input image.

    Parameters
    ----------
    model : torchvision.models
        The neural network.
    image_shape : tuple of ints
        (vertical_dimension, horizontal_dimension) in pixels.

    Returns
    -------
    layers : list of nn.Modules
        The layers ordered according to how the data is propagated through
        the model. Container layers are not included.
    rf_size : list of tuples
        Each tuple contains the height and width of the receptive field that
        corresponds to the layer of the same index in the <layers> list.
    """
    converter = SpatialIndexConverter(model, image_shape)
    rf_sizes = []
    for i in range(len(converter.layers)):
        try:
            # Using the spatial center to avoid boundary effects.
            _, v_size, h_size = converter.output_sizes[i]
            coord = converter.convert((v_size//2, h_size//2), i, 0, is_forward=False)
            rf_size = (coord[2] - coord[0] + 1, coord[3] - coord[1] + 1)
        except:
            # Don't care about 1D layers.
            rf_size = (-1, -1)

        rf_sizes.append(rf_size)
    return converter.layers, rf_sizes


if __name__ == '__main__':
    model = models.alexnet()
    layers, rf_sizes = rf_sizes(model, (227, 227))
    print(rf_sizes)
