"""
The visualization method is inspired by paper "Object Detectors Emerge in Deep
Scene CNNs (2015)" by Zhou et al., Avaliable at: https://arxiv.org/abs/1412.6856

In the paper's section 3.2 "VISUALIZING THE RECEPTIVE FIELDS OF UNITS AND THEIR
ACTIVATION PATTERNS," the authors described their method:

    "... we replicate each image many times with small random occluders (image
    patches of size 11x11) at different locations in the image. Specifically,
    we generate occluders in a dense grid with a stride of 3. This results in
    about 5000 occluded images per original image. We now feed all the occluded
    images into the same network and record the change in activation as
    compared to using the original image. If there is a large discrepancy, we
    know that the given patch is important and vice versa. This allows us to
    build a discrepancy map for each image. Finally, to consolidate the
    information from the K images, we center the discrepancy map around the
    spatial location of the unit that caused the maximum activation for the
    given image. Then we average the re-centered discrepancy maps to generate
    the final RF."

Their code is avaliable at:
https://github.com/zhoubolei/cnnvisualizer/blob/master/pytorch_generate_unitsegments.py

The following Python script also uses occluders to generate discrepancy maps.
However, unlike Zhou et al. (2015), the occluders were only used on the top
and bottom image patches, rather than entire images. Also, instead of using
a fixed size and stride for the occluders, we scaled the occluder size and
stride linearly with the RF field size according to the formula:

    occluder_size = rf_size // 10
    occluder_stride = occluder_size // 3

Tony Fu, Aug 5, 2022
"""
import sys

import torch
import torch.fx as fx
import torch.nn as nn
from torchvision import models

sys.path.append('../..')
import src.rf_mapping.constants as c
from src.rf_mapping.net import get_truncated_model


#######################################.#######################################
#                                                                             #
#                                                          #
#                                                                             #
###############################################################################
