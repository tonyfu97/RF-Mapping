"""
Code for getting information about spatial dimensions and manipulating spatial
indices.

Note: all code assumes that the y-axis points downward.

Tony Fu, July 6th, 2022
"""
import math
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

from hook import SizeInspector, LayerOutputInspector
from image import clip, preprocess_img_to_tensor, tensor_to_img
import constants as c


#######################################.#######################################
#                                                                             #
#                                    CLIP                                     #
#                                                                             #
###############################################################################
def clip(x, x_min, x_max):
    """Limits x to be x_min <= x <= x_max."""
    x = min(x_max, x)
    x = max(x_min, x)
    return x


#######################################.#######################################
#                                                                             #
#                               CALCULATE CENTER                              #
#                                                                             #
###############################################################################
def calculate_center(output_size):
    """
    center = (output_size - 1)//2.
    
    Parameters
    ----------
    output_size : int or (int, int)
        The size of the output maps in (height, width) format.
    
    Returns
    -------
    The index (int) or indices (int, int) of the spatial center.
    """
    if isinstance(output_size, (tuple, list, np.ndarray)):
        if len(output_size) != 2:
            raise ValueError("output_size should have a length of 2.")
        c1 = calculate_center(output_size[0])
        c2 = calculate_center(output_size[1])
        return c1, c2
    else:
        return (output_size - 1)//2


#######################################.#######################################
#                                                                             #
#                           SPATIAL INDEX CONVERTER                           #
#                                                                             #
###############################################################################
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
    image_dir = c.REPO_DIR + '/data/imagenet/0.npy'
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
    image_dir = c.REPO_DIR + '/data/imagenet/0.npy'
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


# if __name__ == '__main__':
#     # _test_forward_conversion() TODO: finish implementing this test
#     _test_backward_conversion()


#######################################.#######################################
#                                                                             #
#                                 GET_RF_SIZES                                #
#                                                                             #
###############################################################################
def get_rf_sizes(model, image_shape, layer_type=nn.Conv2d):
    """
    Find the receptive field (RF) sizes of all layers of the specified type.
    The sizes here are in pixels with respect to the input image.

    Parameters
    ----------
    model : torchvision.models
        The neural network.
    image_shape : (int, int)
        (vertical_dimension, horizontal_dimension) in pixels. This should not
        really matter unless the image_shape is smaller than some of the
        rf_sizes.
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


#######################################.#######################################
#                                                                             #
#                           GET_CONV_OUTPUT_SHAPES                            #
#                                                                             #
###############################################################################
def get_conv_output_shapes(model, image_shape):
    inspector = LayerOutputInspector(model, nn.Conv2d)
    dummy_img = np.zeros((3, image_shape[0], image_shape[1]))
    layer_outputs = inspector.inspect(dummy_img)
    return [layer_output.shape[1:] for layer_output in layer_outputs]


#######################################.#######################################
#                                                                             #
#                                   RF GRID                                   #
#          NOT USED ANYMORE. Use bar.stimset_gridx_barma() instead.           #
#                                                                             #
###############################################################################
class RfGrid:
    """
    A class that divides a given receptive field into equally spaced gris up to
    some rounding error. Useful for stimulus placement.
    """
    def __init__(self, model, image_shape):
        self.model = copy.deepcopy(model)
        self.image_shape = image_shape
        self.converter = SpatialIndexConverter(model, image_shape)
    
    def _divide_from_middle(self, start, end, increment):
        """
        For example, if given min = 15, max = 24, increment = 4:

          -15---16---17---18---19---20---21---22---23---24-

        Divides it into:

              |                   |                   |
          -15-|-16---17---18---19-|-20---21---22---23-|-24-
              |                   |                   |
        
        Then rounds the numbers to the nearest intergers:

                |                   |                   |
          -15---16---17---18---19---20---21---22---23---24-
                |                   |                   |
        
        Returns [16, 20, 24] in this case.
        """
        if math.isclose(increment, 0):
            warnings.warn("The increment is too close to zero. Returns -1")
            return [-1]

        middle = (start + end)/2
        indices = [middle]
        
        # Find indices that are smaller than the middle.
        while (indices[-1] - increment >= start):
            indices.append(indices[-1] - increment)
        indices = indices[::-1]

        # Find indices that are larger than the middle.
        while (indices[-1] + increment <= end):
            indices.append(indices[-1] + increment)

        return [round(i) for i in indices]

    def get_grid_coords(self, layer_idx, spatial_index, grid_spacing):
        """
        Generates a list of coordinates that equally divide the receptive
        field of a unit up to some rounding. The grid is centered at the center
        of the receptive field.

        Parameters
        ----------
        layer_idx : int
            The index of the layer. See 'hook.py' module for details.
        spatial_index : int or (int, int)
            The spatial position of the unit of interest. Either in (y, x)
            format or a flatten index. Not in pixels but should be w.r.t.
            the output maps of the layer.
        grid_spacing : float
            The spacing between the grid lines (pix).

        Returns
        -------
        grid_coords : [(int, int), ...]
            The coordinates of the intersections in [(x0, y0), (x1, y1), ...]
            format.
        """
        # Project the unit backward to the image space.
        y_min, x_min, y_max, x_max = self.converter.convert(spatial_index,
                                                            layer_idx,
                                                            end_layer_index=0,
                                                            is_forward=False)
        x_list = self._divide_from_middle(x_min, x_max, grid_spacing)
        y_list = self._divide_from_middle(y_min, y_max, grid_spacing)
        
        grid_coords = []
        for x in x_list:
            for y in y_list:
                grid_coords.append((x, y))

        return grid_coords


#######################################.#######################################
#                                                                             #
#                               TRUNCATED_MODEL                               #
#                                                                             #
###############################################################################
def truncated_model(x, model, layer_index):
    """
    Returns the output of the specified layer without forward passing to
    the subsequent layers.

    Parameters
    ----------
    x : torch.tensor
        The input. Should have dimension (1, 3, 2xx, 2xx).
    model : torchvision.model.Module
        The neural network (or the layer if in a recursive case).
    layer_index : int
        The index of the layer, the output of which will be returned. The
        indexing excludes container layers.

    Returns
    -------
    y : torch.tensor
        The output of layer with the layer_index.
    layer_index : int
        Used for recursive cases. Should be ignored.
    """
    # If the layer is not a container, forward pass.
    if (len(list(model.children())) == 0):
        return model(x), layer_index - 1
    else:  # Recurse otherwise.
        for sublayer in model.children():
            x, layer_index = truncated_model(x, sublayer, layer_index)
            if layer_index < 0:  # Stop at the specified layer.
                return x, layer_index


#######################################.#######################################
#                                                                             #
#                               XN_TO_CENTER_RF                               #
#                                                                             #
###############################################################################
def xn_to_center_rf(model):
    """
    Return the input image size xn = yn just big enough to center the RF of
    the center units (of all Conv2d layer) in the pixel space. Need this
    function because we don't want to use the full image size (227, 227).

    Parameters
    ----------
    model : torchvision.models
        The neural network.

    Returns
    -------
    xn_list : [int, ...]
        A list of xn (which is also yn since we assume RF to be square). 
    """
    model.eval()
    layer_indices, rf_sizes = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)
    xn_list = []
    
    for layer_index, rf_size in zip(layer_indices, rf_sizes):
        # Set before and after to different values first
        center_response_before = -2
        center_response_after = -1
        rf_size = rf_size[0]
        xn = int(rf_size * 1.1)  # add a 10% padding.

        # If response before and after perturbation are identical, the unit
        # RF is centered.
        while(center_response_before != center_response_after):
            xn += 1
            dummy_input = torch.rand((1, 3, xn, xn))
            y, _ = truncated_model(dummy_input, model, layer_index)
            yc, xc = calculate_center(y.shape[-2:])
            center_response_before = y[0, 0, yc, xc].item()
            
            # Skip this loop if the paddings on two sides aren't equal.
            if ((xn - rf_size)%2 != 0):
                continue

            padding = (xn - rf_size) // 2
            
            # Add perturbation to the surrounding padding.
            dummy_input[:, :,  :padding,  :padding] = 10000
            dummy_input[:, :, -padding:, -padding:] = 10000
            y, _ = truncated_model(dummy_input, model, layer_index)
            center_response_after = y[0, 0, yc, xc].item()

        xn_list.append(xn)
    return xn_list


# Test
if __name__ == '__main__':
    from torchvision.models import AlexNet_Weights, VGG16_Weights
    model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    # model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    print(xn_to_center_rf(model))
