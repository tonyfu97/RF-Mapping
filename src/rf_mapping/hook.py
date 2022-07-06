"""
Functions that register hooks for a variety of purposes.

Tony Fu, Jun 22, 2022
"""
import os
import sys
import copy
import math


import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

sys.path.append('..')
from image import clip, preprocess_img_to_tensor, tensor_to_img
import constants as c


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
        if (isinstance(image, np.ndarray)):
            image = preprocess_img_to_tensor(image)
        _ = self.model(image)
        return self.layer_outputs


def get_conv_output_shapes(model, image_shape):
    inspector = LayerOutputInspector(model, nn.Conv2d)
    dummy_img = np.zeros((3, image_shape[0], image_shape[1]))
    layer_outputs = inspector.inspect(dummy_img)
    return [layer_output.shape[1:] for layer_output in layer_outputs]


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
    repo_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
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


class SpatialIndexConverter(SizeInspector):
    """
    A class containing the model- and image-shape-specific conversion functions
    of the spatial indices across different layers. Useful for receptive field
    mapping and other tasks that involve the mappings of spatial locations
    onto a different layer.
    """
    def __init__(self, model, image_shape):
        """
        Constructs a SpatialIndexConverter object.

        Parameters
        ----------
        model : torchvision.models
            The neural network.
        image_shape : tuple of ints
            (vertical_dimension, horizontal_dimension) in pixels.
        """
        super().__init__(model, image_shape)
        self.dont_need_conversion = (nn.Sequential,
                                    nn.ModuleList,
                                    nn.Sigmoid,
                                    nn.ReLU,
                                    nn.Tanh,
                                    nn.Softmax2d,
                                    nn.BatchNorm2d,
                                    nn.Dropout2d,)
        self.need_convsersion = (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)

    def _forward_transform(self, x_min, x_max, stride, kernel_size, padding, max_size):
        x_min = math.floor((x_min + padding - kernel_size)/stride + 1)
        x_min = clip(x_min, 0, max_size)
        x_max = math.floor((x_max + padding)/stride)
        x_max = clip(x_max, 0, max_size)
        return x_min, x_max

    def _backward_transform(self, x_min, x_max, stride, kernel_size, padding,max_size):
        x_min = (x_min * stride) - padding
        x_min = clip(x_min, 0, max_size)
        x_max = (x_max * stride) + kernel_size - 1 - padding
        x_max = clip(x_max, 0, max_size)
        return x_min, x_max

    def _one_projection(self, layer_index, vx_min, hx_min, vx_max, hx_max, is_forward):
        layer = self.layers[layer_index]

        # Check the layer types to determine if a projection is necessary.
        if isinstance(layer, self.dont_need_conversion):
            return vx_min, hx_min, vx_max, hx_max
        if isinstance(layer, nn.Conv2d) and (layer.dilation != (1,1)):
            raise ValueError("Dilated convolution is currently not supported by SpatialIndexConverter.")
        if isinstance(layer, nn.MaxPool2d) and (layer.dilation != 1):
            raise ValueError("Dilated max pooling is currently not supported by SpatialIndexConverter.")

        # Use a different max size and transformation function depending on the
        # projection direction.
        if is_forward:
            _, v_max_size, h_max_size = self.output_sizes[layer_index]
            transform = self._forward_transform
        else:
            _, v_max_size, h_max_size = self.input_sizes[layer_index]
            transform = self._backward_transform

        if isinstance(layer, self.need_convsersion):
            try:
                vx_min, vx_max = transform(vx_min, vx_max,
                                        layer.stride[0], layer.kernel_size[0],
                                        layer.padding[0], v_max_size)
                hx_min, hx_max = transform(hx_min, hx_max,
                                        layer.stride[1], layer.kernel_size[1],
                                        layer.padding[1], h_max_size)
                return vx_min, hx_min, vx_max, hx_max
            except:
                # Sometimes the layer attributes do not come in the form of
                # a tuple. 
                vx_min, vx_max = transform(vx_min, vx_max,
                                        layer.stride, layer.kernel_size,
                                        layer.padding, v_max_size)
                hx_min, hx_max = transform(hx_min, hx_max,
                                        layer.stride, layer.kernel_size,
                                        layer.padding, h_max_size)
                return vx_min, hx_min, vx_max, hx_max

        raise ValueError(f"{type(layer)} is currently not supported by SpatialIndexConverter.")

    def _process_index(self, index, start_layer_index):
        """
        Make sure that the index is a tuple of two indices. Unravel from 1D
        to 2D indexing if necessary.

        Returns
        -------
        index : tuple of ints
            The spatial coordinates of the point of interest in 
            (vertical index, horizontal index) format. Note that the vertical
            index increases downward.
        """
        try:
            if len(index)==2:
                return index
            else:
                raise Exception
        
        except:
            _, output_height, output_width = self.output_sizes[start_layer_index]
            return np.unravel_index(index, (output_height, output_width))

        

    def convert(self, index, start_layer_index, end_layer_index, is_forward):
        """
        Converts the spatial index across layers. Given a spatial location, the
        method returns a "box" in (vx_min, hx_min, vx_max, hx_max) format.

        Parameters
        ----------
        index : int or tuple of two ints
            The spatial index of interest. If only one int is provided, the
            function automatically unravel it accoording to the image's shape.
        start_layer_index : int
            The index of the starting layer. If you are not sure what index
            to use, call the .print_summary() method.
        end_layer_index : int
            The index of the destination layer. If you are not sure what index
            to use, call the .print_summary() method.
        is_forward : bool
            Is it a forward projection or backward projection. See below.

        -----------------------------------------------------------------------

        If is_forward == True, then forward projection:
        Projects from the INPUT of the start_layer to the OUTPUT of the
        end_layer:
                           start_layer                end_layer
                       -----------------          ----------------
                               |                         |
                         input |  output    ...    input |  output
                               |                         |
                       -----------------          ----------------     
        projection:        * --------------------------------->

        -----------------------------------------------------------------------

        If is_forward == False, then backward projection:
        Projects from the OUTPUT of the start_layer to the INPUT of the
        end_layer:
                            end_layer                start_layer
                       -----------------          ----------------
                               |                         |
                         input |  output    ...    input |  output
                               |                         |
                       -----------------          ----------------  
        projection:         <-------------------------------- *

        -----------------------------------------------------------------------

        This means that it is totally legal to have a start_layer_index that
        is equal to the end_layer_index. For example, the code:

            converter = SpatialIndexConverter(model, (227, 227))
            coord = converter.convert((111,111), 0, 0, is_forward=True)

        will return the coordinates of a box that include all output points
        of layer no.0 that can be influenced by the input pixel at (111,111).
        On the other hand, the code:

            coord = converter.convert((28,28), 0, 0, is_forward=False)

        will return the coordinates of a box that include all input pixels
        that can influence the output (of layer no.0) at (28,28).
        """
        vx, hx = self._process_index(index, start_layer_index)
        vx_min, vx_max = vx, vx
        hx_min, hx_max = hx, hx

        if is_forward:
            index_gen = range(start_layer_index, end_layer_index + 1)
        else:
            index_gen = range(start_layer_index, end_layer_index - 1, -1)

        for layer_index in index_gen:
            vx_min, hx_min, vx_max, hx_max = self._one_projection(layer_index, 
                                   vx_min, hx_min, vx_max, hx_max, is_forward)

        return vx_min, hx_min, vx_max, hx_max


if __name__ == '__main__':
    model = models.alexnet()
    converter = SpatialIndexConverter(model, (227, 227))
    coord = converter.convert((0, 0), 0, 0, is_forward=False)
    print(coord)


def _test_forward_conversion():
    """
    Test function for the SpatialIndexConverter class when is_forward is set to
    True, i.e., forward projection from the input of a shallower layer onto
    the output of a deeper layer. This function test whether all four corners
    of the box returned by the forward projection indeed can be influenced by
    pertibation the starting point.
    """
    import random
    # Parameters: You are welcomed to change them.
    model = models.alexnet(pretrained=True)
    model.eval()
    image_size = (198, 203)
    start_index = (random.randint(0, image_size[0]-1),
                   random.randint(0, image_size[1]-1))
    
    # Get an arbitrary image.
    repo_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    image_dir = f"{repo_dir}/data/imagenet/0.npy"
    test_image = np.load(image_dir)
    test_image_tensor = preprocess_img_to_tensor(test_image, image_size)
    
    # Create an identical image but the start index is set to zero.
    image_diff_point = test_image_tensor.detach().clone()
    image_diff_point[:, :, start_index[0], start_index[1]] = 0
    
    # Aliases to avoid confusion in the for loop.
    x = test_image_tensor.detach().clone()
    x_diff_point = image_diff_point.detach().clone()

    # The Converter object to be tested.
    converter = SpatialIndexConverter(model, image_size)
    
    try:
        # TODO: finish implementing the test.
        for layer_i, output_size in enumerate(converter.output_sizes):
            _, max_height, max_width = output_size
            
            vx_min, hx_min, vx_max, hx_max = converter.convert(start_index,
                                                0, layer_i, is_forward=True)

            # Forward-pass the two images to the current layer of interest.
            for l in converter.layers[:layer_i + 1]:
                x = l.forward(x)
                x_rand_outside = l.forward(x_rand_outside)

    except:
        print(f"The rest of the layers are probably 1D.")

def _test_backward_conversion():
    """
    Test function for the SpatialIndexConverter class when is_forward is set to
    False, i.e., backward projection from the output of a deeper layer onto
    the input of a shallower layer. The test can be summarized as follows:
        1. Starting from the first layer, choose a point in its output space.
        2. Call the SpatialIndexConverter to back-project the point back onto
           the pixel space. The class will return a the indices of the RF.
        3. Get an arbitrary image, then create a copy image that is identical
           inside the RF, but are random values outside of the RF.
        4. Present the two images to the model, and test if the responses at
           the point in the output space are the same. If so, then the
           backward index conversion is deemed successful.
    """
    # Parameters: You are welcomed to change them.
    model = models.vgg16(pretrained=True)
    model.eval()
    image_size = (227, 227)
    show = True

    # Get an arbitrary image.
    repo_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
    image_dir = f"{repo_dir}/data/imagenet/0.npy"
    test_image = np.load(image_dir)
    test_image_tensor = preprocess_img_to_tensor(test_image, image_size)

    if show:
        plt.figure()
        plt.imshow(tensor_to_img(test_image_tensor))
        plt.title(f"Test image")
        plt.show()

    # The Converter object to be tested.
    converter = SpatialIndexConverter(model, image_size)

    for layer_i in range(len(converter.layers)):
        try:
            # Use the center point of the layer output as starting point for
            # back projection. You are welcome to change this.
            _, max_height, max_width = converter.output_sizes[layer_i]
            index = (max_height//2, max_width//2)
            
            # Get the RF box that should contain all the points in that can
            # influence the output at the index specified above.
            vx_min, hx_min, vx_max, hx_max = converter.convert(index, 
                                                layer_i, 0, is_forward=False)
            rf_size = (vx_max - vx_min + 1, hx_max - hx_min + 1)
            
            # Create an identical image but its pixels outside of the RF are
            # replaced with random values.
            image_rand_outside = torch.rand(test_image_tensor.shape)
            image_rand_outside[:,:,vx_min:vx_max+1, hx_min:hx_max+1] =\
                    test_image_tensor.detach().clone()[:,:,vx_min:vx_max+1, hx_min:hx_max+1]

            # Aliases to avoid confusion in the for loop.
            x = test_image_tensor.detach().clone()
            x_rand_outside = image_rand_outside.detach().clone()

            # Forward-pass the two images to the current layer of interest.
            for l in converter.layers[:layer_i + 1]:
                x = l.forward(x)
                x_rand_outside = l.forward(x_rand_outside)

            # Backward conversion test: Check if the responses at the output index
            # are the same for the two images.
            result = torch.eq(x[:, :, index[0], index[1]], 
                            x_rand_outside[:, :, index[0], index[1]]).numpy()
            if (np.sum(result) != result.shape[1]):
                print(f"Backward conversion failed for layer no.{layer_i}.")
            else:
                print(f"Backward conversion is successful for layer no.{layer_i}.")

            # Plot the two images used.
            if show:
                plt.figure(figsize=(6,3))
                plt.suptitle(f"Layer no.{layer_i}, RF size = {rf_size}")  
                
                plt.subplot(1, 2, 1)
                plt.imshow(tensor_to_img(test_image_tensor))
                plt.title(f"Original")
                
                plt.subplot(1, 2, 2)
                plt.imshow(tensor_to_img(image_rand_outside))
                plt.title(f"Rand outside")

                plt.show()
        except:
            print(f"The rest of the layers from layer no.{layer_i} are probably 1D.")
            break


if __name__ == '__main__':
    # _test_forward_conversion()
    _test_backward_conversion()


def get_rf_sizes(model, image_shape, layer_type=nn.Conv2d):
    """
    Find the receptive field (RF) sizes of all layers of the specified type.
    The sizes here are in pixels with respect to the input image.

    Parameters
    ----------
    model : torchvision.models
        The neural network.
    image_shape : (int, int)
        (vertical_dimension, horizontal_dimension) in pixels.
    layer_type: nn.Module
        The type of layer to consider.

    Returns
    -------
    layer_indices : [int, ...]
        The indices of the layers of the specified layer type.
    rf_size : [(int, int), ...]
        Each tuple contains the height and width of the receptive field that
        corresponds to the layer of the same index in the <layers> list.
    """
    converter = SpatialIndexConverter(model, image_shape)
    layer_indices = []
    rf_sizes = []
    for i, layer in enumerate(converter.layers):
        if isinstance(layer, layer_type):
            layer_indices.append(i)
            # Using the spatial center to avoid boundary effects.
            _, v_size, h_size = converter.output_sizes[i]
            coord = converter.convert((v_size//2, h_size//2), i, 0, is_forward=False)
            rf_size = (coord[2] - coord[0] + 1, coord[3] - coord[1] + 1)
            rf_sizes.append(rf_size)
        
    return layer_indices, rf_sizes


if __name__ == '__main__':
    model = models.alexnet()
    layer_indices, rf_sizes = get_rf_sizes(model, (227, 227))
    print(layer_indices)
    print(rf_sizes)