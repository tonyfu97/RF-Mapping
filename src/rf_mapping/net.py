import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.append('../..')
import src.rf_mapping.constants as c


model = models.inception_v3()
from torchvision.models.feature_extraction import get_graph_node_names
train_nodes, eval_nodes = get_graph_node_names(model, suppress_diff_warning=True)
print(eval_nodes)


#######################################.#######################################
#                                                                             #
#                                 IS_RESIDUAL                                 #
#                                                                             #
###############################################################################
def is_residual(container_layer):
    """Check if the container layer has residual connection or not."""
    has_conv = False
    has_conv1x1 = False
    first_in_channels = None

    for sublayer in container_layer.children():
        if isinstance(sublayer, nn.Conv2d):
            if (not has_conv):
                has_conv = True
                first_in_channels = sublayer.in_channels
            
            if (not has_conv1x1):
                try:
                    has_conv1x1 = (sublayer.kernel_size == (1,1))
                except:
                    has_conv1x1 = (sublayer.kernel_size == 1)

    if not has_conv:
        return 0

    dummy_input = torch.ones((1, first_in_channels, 100, 100)).to(c.DEVICE)
    x = dummy_input.detach()
    for sublayer in container_layer.children():
        x = sublayer(x)
    print(x.mean())
    print(container_layer(dummy_input).mean())
    print(has_conv1x1)

    original_output = container_layer(dummy_input)
    if not has_conv1x1:
        if torch.sum(x != original_output) == 0:
            return 0
        
        rectified_x = x.detach()
        rectified_x += dummy_input
        rectified_x[rectified_x < 0] = 0
        if torch.sum(rectified_x != original_output) == 0:
            return 1
        print(rectified_x.mean())

    conv1x1 = nn.Conv2d(x.shape[1], original_output.shape[1], kernel_size=1, stride=2, bias=False)
    subsampled_x = conv1x1(x)
    subsampled_x += dummy_input
    subsampled_x[subsampled_x < 0] = 0
    if torch.sum(subsampled_x != original_output) == 0:
        return 2

res_block = list(list(models.resnet18().children())[4].children())[0]
print(is_residual(res_block))

not_res_block = list(models.alexnet().children())[0]
print(is_residual(not_res_block))

not_res_block2 = nn.Sequential()
print(is_residual(not_res_block2))


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