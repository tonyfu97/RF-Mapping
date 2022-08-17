import os
import sys
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append('../..')
from src.rf_mapping.hook import SizeInspector
from src.rf_mapping.spatial import get_rf_sizes, SpatialIndexConverter
import src.rf_mapping.constants as c
from src.rf_mapping.net import get_truncated_model
from src.rf_mapping.image import (preprocess_img_to_tensor,
                                  preprocess_img_for_plot,
                                  make_box,)


#######################################.#######################################
#                                                                             #
#                               GUIDED_BACKPROP                               #
#                                                                             #
###############################################################################
class GuidedBackprop:
    """
    Generates the guided backpropagation gradient maps for a spatial location
    of a particular unit of a particular layer.
    
    This class was implemented by utkuozbulak (https://github.com/utkuozbulak/
    pytorch-cnn-visualizations) and later modified by Dr. Taekjun Kim to
    include unit specificity and by me to improve generalizability for other
    model architectures.
    """
    def __init__(self, model, layer_index):
        self.model = get_truncated_model(model, layer_index)

        self.gradients = None
        self.forward_relu_outputs = []

        self._register_hook_to_first_layer(self.model)
        self._update_relus(self.model)
        
    def _first_hook_function(self, module, grad_in, grad_out):
        self.gradients = grad_in[0]
        # [0] bc we are only interested in one unit at a time, so grad_in
        # will be a tuple of size 1.
        if not isinstance(module, nn.Conv2d):
            warnings.warn("The first layer is not Conv2d.")

    def _forward_hook_function(self, module, ten_in, ten_out):
        """
        Stores results of forward pass.
        """
        if (isinstance(module, nn.ReLU)):
            self.forward_relu_outputs.append(ten_out)

    def _backward_hook_function(self, module, grad_in, grad_out):
        """
        Rectifies gradients.
        """
        if (isinstance(module, nn.ReLU)):
            # print(layer_name)
            target_grad = grad_in[0]
            # [0] bc we are only interested in one unit at a time, so grad_in
            # will be a tuple of size 1.
            
            # Get last forward output of ReLU. Use non-zero values to create
            # a mask of 1's and 0's.
            try:
                corresponding_forward_output = self.forward_relu_outputs[-1]
                corresponding_forward_output[corresponding_forward_output > 0] = 1
                # Rectification (see Springenberg et al. 2015 Figure 1).
                modified_grad_out = corresponding_forward_output * F.relu(target_grad)
            except:
                # TODO: This is a temporary fix for resnet18().
                # Why this work: the original error (in the 'try' clause) is
                # because, for the shortcut conv layers, the target_grad comes
                # from the ReLU between the residual blocks, while the
                # corresponding_forward_output comes from the ReLU inside the
                # residual block that runs parallel to the shortcut. This is
                # because the network calculates the residual path first, then
                # calculates the shortcut path. So, even though the backprop
                # did not require the residual path, its ReLU is calculated and
                # appended to the self.forward_relu_outputs list.
                del self.forward_relu_outputs[-1]

                corresponding_forward_output = self.forward_relu_outputs[-1]
                corresponding_forward_output[corresponding_forward_output > 0] = 1
                # Rectification (see Springenberg et al. 2015 Figure 1).
                modified_grad_out = corresponding_forward_output * F.relu(target_grad)

            # Remove last forward output and return.
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

    def _register_hook_to_first_layer(self, layer):
        # Skip any container layers.
        while (len(list(layer.children())) != 0):
            layer = list(layer.children())[0]
        layer.register_backward_hook(self._first_hook_function)

    def _update_relus(self, layer):
        """
        Updates ReLU activation functions so that they now:
            1- rectify gradient values so that there's no negative gradients.
            2- store output in forward pass.
        """
        # If layer is not a container, register hook.
        if (len(list(layer.children())) == 0):
            # self.layers.append(layer)  # Keep track of all non-container layers
            layer.register_forward_hook(self._forward_hook_function)
            layer.register_backward_hook(self._backward_hook_function)

        # Otherwise (i.e.,the layer is a container type layer), recurse.
        else:
            for i, sublayer in enumerate(layer.children()):
                self._update_relus(sublayer)

    def generate_gradients(self, img, target_unit, target_spatial_idx):
        """
        Generates the gradient map of the target with respect to the image.

        Parameters
        ----------
        img : numpy.array
            The input image.
        # target_layer : int
        #     The index of the target layer. Note that the indexing here does not
        #     include container layers.
        target_unit : int
            The id of the unit. 
        target_spatial_idx : int or (int, int)
            The spatial index of the target location on the output feature map.
            If only one scalar is provided, this function unravels it into 2D
            index.

        Returns
        -------
        gradient_img : numpy.array
            The gradient map.
        """
        self.model.zero_grad()

        # Forward pass.
        x = preprocess_img_to_tensor(img).clone().detach().requires_grad_(True)
        x = self.model(x)

        # We only care about the gradient w.r.t. the target. 
        if not isinstance(target_spatial_idx, (tuple, list)):
            target_spatial_idx = np.unravel_index(target_spatial_idx, (x.shape[2], x.shape[3]))
        x_target_only = torch.zeros(x.shape, dtype=torch.float).to(c.DEVICE)
        x_target_only[0, target_unit, target_spatial_idx[0], target_spatial_idx[1]] =\
                        x[0, target_unit, target_spatial_idx[0], target_spatial_idx[1]]
        x.backward(gradient=x_target_only)

        if self.gradients is None:
            raise ValueError("Target layer must be Conv2d.")

        # [0] to get rid of the first channel (1, 3, 22x, 22x).
        gradients_img = self.gradients.data.cpu().numpy()[0]
        return gradients_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    model = models.alexnet(pretrained = True).to(c.DEVICE)
    # model = models.resnet18(pretrained = True).to(c.DEVICE)

    inspector = SizeInspector(model, (227, 227))
    # inspector.print_summary()

    img_dir = c.REPO_DIR + '/data/imagenet'
    img_idx = 5
    img_path = os.path.join(img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    img = preprocess_img_for_plot(img)

    layer_idx = 3  # AlexNet = [0, 3, 6, 8, 10] for conv1-5
    unit_idx = 2
    spatial_idx = (5, 5)
    gbp = GuidedBackprop(model, layer_idx)
    gbp_map = gbp.generate_gradients(img, unit_idx, spatial_idx)

    plt.figure(figsize=(10,15))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(preprocess_img_for_plot(gbp_map))
    plt.show()


#######################################.#######################################
#                                                                             #
#                             TEST_GUIDED_BACKPROP                            #
#                                                                             #
###############################################################################
def _test_guided_backprop():
    # model = models.resnet18(pretrained=True)
    model = models.alexnet(pretrained=True)
    unit_idx = 1
    image_size = (227, 227)
    layer_indices, rf_sizes = get_rf_sizes(model, image_size, nn.Conv2d)

    output_sizes = SizeInspector(model, image_size).output_sizes
    converter = SpatialIndexConverter(model, image_size)

    img_idx = 5
    img_path = os.path.join(c.REPO_DIR, img_dir, f"{img_idx}.npy")
    img = np.load(img_path)
    # dummy_img = preprocess_img_for_plot(img)
    dummy_img = np.random.rand(3,227,227)
    # dummy_img = np.ones((3, 227, 227)) * 100

    def img_proc(img):
        vmax = img.max()
        vmin = img.min()
        img = (img - vmin)/(vmax - vmin)
        return np.transpose(img, (1,2,0))
    
    for conv_i, layer_idx in enumerate(layer_indices):
        output_size = output_sizes[layer_idx][-1]
        rf_size = rf_sizes[conv_i][0]
        spatial_idx = ((output_size-1)//2, (output_size-1)//2)

        gbp = GuidedBackprop(model, layer_idx)
        gbp_map = gbp.generate_gradients(dummy_img, unit_idx, spatial_idx)
        
        box = converter.convert(spatial_idx, layer_idx, 0, is_forward=False)
        
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"guided backprop of conv{conv_i+1} (RF = {rf_size}, output_size = {output_size})")

        plt.subplot(1,2,1)
        plt.imshow(img_proc(gbp_map))
        plt.title(f"array of ones")
        rect = make_box(box)
        ax = plt.gca()
        ax.add_patch(rect)

        print(gbp_map.shape)
        print(gbp_map.max(), gbp_map.min())
        plt.subplot(1,2,2)
        plt.imshow((np.mean(gbp_map, axis=0) != 0), cmap='gray')
        plt.title(f"binarized (non-zeros = white)")
        rect = make_box(box)
        ax = plt.gca()
        ax.add_patch(rect)
        
        plt.grid()
        plt.show()


if __name__ == "__main__":
    _test_guided_backprop()

"""
Test observations:

(1) When the input image is an array of ones, i.e., 
                dummy_img = torch.ones((1,3,227,227))
    the binarized gradient maps show that non-zero gradients are distributed at
    the top-left corner of the RF and failed to fill the entire RF. Instead,
    the area of the non-zero gradient is roughly equal to the RF of an early
    layer. This suggests that gradient calculations may systemically bias the
    top-left. [Need math proof]

(2) When the input image is natural, the non-zero gradients fill the entire
    RF.

(3) When the RF of the unit is close to or larger than the image size, the left
    and upper edge of the RF is often cropped off. This is because many of
    those layers (e.g. conv16-20 of resnet18) have a feature map size that is
    an even number. For exmample, the center unit of the feature map size of
    8 is (8 - 1)//2 = 3, a little bit to the left of the actual center, and
    since the RF is so large in deeper layers, a little bit off-center will
    result in a big shift from the actual center on the image.
"""
