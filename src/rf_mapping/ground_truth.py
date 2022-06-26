"""
Code to find the image patches that drive the units the most/least. 

Tony Fu, Jun 25, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


from hook import LayerOutputInspector
from image import ImageNetDataset


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained=True).to(device)
num_images = 100
img_dir = Path(__file__).parent.parent.parent.joinpath('data/imagenet')
img_names = [f"{i}.npy" for i in range(num_images)]

imagenet_data = ImageNetDataset(img_dir, img_names)
imagenet_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=False)

inspector = LayerOutputInspector(model, layer_types=(nn.Conv2d))

for batch, _ in imagenet_dataloader:
    inspector.inspect(batch[0].to(device))
