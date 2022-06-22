"""
Code to find the natural image patches that drive the units the most/least.

Tony Fu, June 18, 2022
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def layer_indices(model, layer_type=nn.Conv2d):
    """
    Recursively find all layers of the type <layer_type> in a given model, then
    returns their indicies of the model. All Pytorch neural networks are tree
    data structures: the "model" is the root node, the container layers such as
    Sequential layers are the intermediate nodes, and layers like Conv2d, ReLU,
    MaxPool2d, and Linear are leave nodes.

    Parameters
    ----------
    model: torchvision.models or torch.nn.Module
        The neural network model or layer object.
    layer_type: type
        The target type of the leave node, e.g., nn.Conv2d, nn.ReLU, etc.

    Returns
    -------
    indicies: list of lists
        Each sublist is a sequence of child indicies from the "model" root node
        to a target layer. For example, if the indicies returned for layer_type
        nn.Conv2d is [0, [0, 3], 1, []], then the first nn.Conv2d layer can be
        accessed with the code:
            list(list(model.children())[0].children())[0]
        And the second nn.Conv2d layer of the model can be found using:
            list(list(model.children())[0].children())[3]
        The 1, [] in the example output means that the second child of the
        model has been checked, and there is no nn.Conv2d in it. There are only
        two nn.Conv2d layers in this example model.
    """
    indicies = _layer_indices(model, layer_type, [])

    # Remove trailing comma.
    indicies[-1] = indicies[-1].split(',')[0]

    return indicies


def _layer_indices(layer, layer_type, indicies):
    """
    Private function used for recursion in layer_indicies().
    """
    # Return the index if layer is a leave node and match the target type.
    if (len(list(layer.children())) == 0):
        if (not isinstance(layer, layer_type)):
            indicies.pop(-1)
        return indicies

    # Recurse otherwise.
    else:
        indicies.append("[")
        for i, sublayer in enumerate(layer.children()):
            indicies.append(f"{i}, ")
            indicies = _layer_indices(sublayer, layer_type, indicies)

        # Remove trailing comma.
        indicies[-1] = indicies[-1].split(',')[0]

        indicies.append("], ")
    return indicies


# def test_layer_indicies(model, layer_type):


if __name__ == '__main__':
    model = models.alexnet()
    print(''.join(layer_indices(model)))
