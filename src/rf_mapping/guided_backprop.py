import os
import sys
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from torchvision import models
import torch.nn.functional as F

sys.path.append('../..')
from src.rf_mapping.hook import SizeInspector
from src.rf_mapping.image import preprocess_img_to_tensor, preprocess_img_for_plot
import src.rf_mapping.constants as c
from src.rf_mapping.net import get_truncated_model


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
                # appended to the self.forward_relu_outputs list nonetheless.
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
