import sys

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
from torchvision import models

sys.path.append('../..')
import src.rf_mapping.constants as c


# TODO:
# 1. Implement network dissection: https://github.com/zhoubolei/cnnvisualizer/blob/master/pytorch_generate_unitsegments.py
#######################################.#######################################
#                                                                             #
#                             GET_TRUNCATED_MODEL                             #
#                                                                             #
###############################################################################
def get_truncated_model(model, layer_index):
    """
    Create a truncated version of the neural network.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network to be truncated.
    layer_index : int
        The last layer (inclusive) to be included in the truncated model.

    Returns
    -------
    truncated_model : torch.nn.Module
        The truncated model.

    Example
    -------
    model = models.alexnet(pretrained=True)
    model_to_conv2 = get_truncated_model(model, 3)
    y = model(torch.ones(1,3,200,200))
    """
    graph = fx.Tracer().trace(model.eval())
    new_graph = fx.Graph()
    layer_counter = 0
    value_remap = {}

    for node in graph.nodes:
        # Create a new module that will be returned
        # new_graph.create_node(node.op, node.target, args=node.args,
        #                       kwargs=node.kwargs, name=node.name)
        value_remap[node] = new_graph.node_copy(node, lambda n : value_remap[n])

        # If the node is a module...
        if node.op == 'call_module':
            # Get the layer object using the node.target attribute.
            layer = model
            for level in node.target.split("."):
                layer = getattr(layer, level)
            # Stop at the desired layer (i.e., truncate).
            if layer_counter == layer_index:
                new_graph.output(node)
                break

            layer_counter += 1

    # new_graph.lint()
    # new_graph.eliminate_dead_code()
    return fx.GraphModule(model, new_graph)
    
if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    dummy_input = torch.ones(1,3,200,200)
    tm = get_truncated_model(model, 100)
    torch.testing.assert_allclose(tm(dummy_input), model(dummy_input))


#######################################.#######################################
#                                                                             #
#                                  LayerNode                                  #
#                                                                             #
###############################################################################
class LayerNode:
    def __init__(self, name, layer=None, parents=(), children=(), idx=None):
        self.idx = idx
        self.name = name
        self.layer = layer
        self.parents = parents
        self.children = children

    def __repr__(self):
        return f"LayerNode '{self.name}', layer = {self.layer}, idx = {self.idx}, "\
               f"parents = {self.parents}, children = {self.children}\n"


#######################################.#######################################
#                                                                             #
#                                 MAKE_GRAPH                                  #
#                                                                             #
###############################################################################
def make_graph(truncated_model):
    # Make sure that the truncated_model is a GraphModule. 
    if not isinstance(truncated_model, fx.graph_module.GraphModule):
        graph = fx.Tracer().trace(truncated_model.eval())
        truncated_model = fx.GraphModule(truncated_model, graph)

    nodes = {}
    idx_count = 0
    # Populate the nodes dictionary with the initialized Nodes.
    for node in truncated_model.graph.nodes:
        # Get the layer torch.nn object.
        if node.op == 'call_module':
            layer = truncated_model
            idx = idx_count
            idx_count += 1
            for level in node.target.split("."):
                layer = getattr(layer, level)
        else:
            layer = None
            idx = None
        # Get the name of the parents.
        parents = []
        for parent in node.args:
            if isinstance(parent, fx.node.Node):
                parents.append(parent.name)
        # Initialize Nodes.
        nodes[node.name] = LayerNode(node.name, layer, parents=tuple(parents),
                                     idx=idx)

    # Determine the children of the nodes.
    for node in truncated_model.graph.nodes:
        for parent in nodes[node.name].parents:
            existing_children = nodes[parent].children
            nodes[parent].children = (*existing_children, node.name)
    return nodes

if __name__ == '__main__':
    model = models.alexnet()
    print(make_graph(model))


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


if __name__ == "__main__":
    res_block = list(list(models.resnet18().children())[4].children())[0]
    print(is_residual(res_block))

    not_res_block = list(models.alexnet().children())[0]
    print(is_residual(not_res_block))

    not_res_block2 = nn.Sequential()
    print(is_residual(not_res_block2))
