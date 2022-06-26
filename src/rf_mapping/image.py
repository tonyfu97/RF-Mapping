"""
Code to for image processing, etc.

Tony Fu, Jun 25, 2022
"""
import os


import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def clip(x, x_min, x_max):
    """Limits x to be x_min <= x <= x_max."""
    x = min(x_max, x)
    x = max(x_min, x)
    return x


def normalize_img(img):
    """Normalize pixel values to be between [-1, 1]."""
    norm_img = img - img.min()
    norm_img = norm_img/norm_img.max()
    return norm_img


def preprocess_img(img, img_size=None):
    """Preprocess image before presenting it to a Pytorch model."""
    norm_img = normalize_img(img)
    norm_img = np.expand_dims(norm_img, axis=0)
    img_tensor = torch.from_numpy(norm_img).type('torch.FloatTensor')

    if img_size is not None:
        resize = T.Resize(img_size)
        img_tensor = resize(img_tensor)

    return img_tensor


def tensor_to_img(img_tensor):
    img = img_tensor.clone().detach()

    if len(img.shape) == 3 and img.shape[1] == 3:
        return np.transpose(torch.squeeze(img),(1,2,0))
    else:
        return img[0,0,...].numpy()


class ImageNetDataset(Dataset):
    def __init__(self, img_dir, img_names):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = preprocess_img
        
    def transform(img):
        return torch.from_numpy(img).type('torch.FloatTensor')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.load(img_path)
        img_tensor = self.transform(img)
        label = 0  # Don't care about labels.

        return img_tensor, label


if __name__ == "__main__":
    num_images = 100
    img_dir = Path(__file__).parent.parent.parent.joinpath('data/imagenet')
    img_names = [f"{i}.npy" for i in range(num_images)]
    imagenet_data = ImageNetDataset(img_dir, img_names)
    imagenet_dataloader = DataLoader(imagenet_data, batch_size=64, shuffle=False)
    imgs, _ = next(iter(imagenet_dataloader))
    img = np.transpose(imgs[63].squeeze(), (1,2,0))
    plt.imshow(img)
    plt.show()
