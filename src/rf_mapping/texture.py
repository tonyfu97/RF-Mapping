

import os
import sys

from PIL import Image
import numpy as np
from torchvision import models
# from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt

sys.path.append('../..')
import src.rf_mapping.constants as c


texture_dir = c.TEXTURE_DIR

texture_types = os.listdir(texture_dir)
for texture_type in texture_types:
    if texture_type.startswith('.'):  # Remove hidden files
        texture_types.remove(texture_type)


for texture_type in texture_types:
    this_texture_type_dir = os.path.join(c.TEXTURE_DIR, texture_type)
    this_textures = os.listdir(this_texture_type_dir)
    
    for this_texture in this_textures:
        this_texture_dir = os.path.join(this_texture_type_dir, this_texture)
        im = np.array(Image.open(this_texture_dir))
        plt.imshow(im)
        plt.title(texture_type)
        plt.show()
        

