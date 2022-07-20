"""
Functions that register hooks for a variety of purposes.

Tony Fu, Jun 22, 2022
"""
import sys
import copy


import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.append('..')
from image import preprocess_img_to_tensor
import constants as c


#######################################.#######################################
#                                                                             #
#                             HOOK FUNCTION BASE                              #
#                                                                             #
###############################################################################
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
            for sublayer in layer.children():
                self.register_forward_hook_to_layers(sublayer)


#######################################.#######################################
#                                                                             #
#                             LAYER OUPUT INSPECTOR                           #
#                                                                             #
###############################################################################
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
        if (isinstance(image, np.ndarray)):
            image = preprocess_img_to_tensor(image)
        _ = self.model(image)
        return self.layer_outputs


#######################################.#######################################
#                                                                             #
#                              CONV UNIT COUNTER                              #
#                                                                             #
###############################################################################
class ConvUnitCounter(HookFunctionBase):
    """
    A class that counts the number of unique kernels of each convolutional
    layers of the given model.
    """
    def __init__(self, model):
        super().__init__(model, nn.Module)
        self._layer_counter = 0
        self.layer_indices = []
        self.num_units = [] 
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        if isinstance(module, nn.Conv2d):
            self.layer_indices.append(self._layer_counter)
            self.num_units.append(module.out_channels)
        self._layer_counter += 1

    def count(self):
        """
        Returns
        -------
        layer_indices : [int, ...]
            The indices of nn.Conv2d layers. For torchvision.model.alexnet(),
            this will be [0, 3, 6, 8, 10].
        num_units : [int, ...]
            The count of unique kernels of each convoltional layers.
            
        layer_indices and num_units have the same length, and their elements
        correspond to each other.
        """
        # Forward pass.
        dummy_input = torch.zeros((1, 3, 227, 227)).to(c.DEVICE)
        self.model(dummy_input)
        
        return self.layer_indices, self.num_units
    

if __name__ == "__main__":
    model = models.alexnet()
    counter = ConvUnitCounter(model)
    layer_indices, num_units = counter.count()
    print(layer_indices)
    print(num_units)


#######################################.#######################################
#                                                                             #
#                              CONV MAX INSPECTOR                             #
#                                                                             #
###############################################################################
class ConvMaxInspector(HookFunctionBase):
    """
    A class that get the maximum activations and indices of all unique
    convolutional kernels, one image at a time.
    """
    def __init__(self, model):
        super().__init__(model, nn.Conv2d)
        self.all_max_activations = []
        self.all_max_indices = []
        self.register_forward_hook_to_layers(self.model)

    def hook_function(self, module, ten_in, ten_out):
        layer_max_activations = []
        layer_max_indices = []
        
        for unit in range(ten_out.shape[1]):
            layer_max_activations.append(ten_out[0,unit,:,:].max().item())
            layer_max_indices.append(ten_out[0,unit,:,:].max().item())
            
        self.all_max_activations.append(layer_max_activations)
        self.all_max_indices.append(layer_max_indices)

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
        copy_indices = self.all_max_indices[:]
        
        self.all_max_activations = []
        self.all_max_indices = []
        
        return copy_activations, copy_indices


def top_bottom_N_image_patches(model, layer_type, image_dir, image_names):
    model.eval()
    inspector = LayerOutputInspector(model, layer_type)
    image = np.load(f"{image_dir}/{image_names[0]}")
    layer_outputs = inspector.inspect(image)
    for layer_output in layer_outputs: 
        print(layer_output.max())


if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    image_dir = c.REPO_DIR + "/data/imagenet"
    image_names = ["0.npy"]
    top_bottom_N_image_patches(model, nn.Conv2d, image_dir, image_names)


#######################################.#######################################
#                                                                             #
#                               SIZE INSPECTOR                                #
#                                                                             #
###############################################################################
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
        self.image_shape = image_shape
        self.layers = []
        self.input_sizes = []
        self.output_sizes = []
        self.register_forward_hook_to_layers(self.model)
        
        self.model(torch.zeros((1,3,*image_shape)).to(c.DEVICE))

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
