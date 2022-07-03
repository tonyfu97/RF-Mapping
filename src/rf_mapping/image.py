"""
Code to for image processing, etc.

Tony Fu, Jun 25, 2022
"""
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    """Normalizes pixel values to be roughly Norm(0, 1)."""
    norm_img = img - img.mean()
    if not math.isclose(norm_img.std(), 0):
        norm_img = norm_img/norm_img.std()
    return norm_img


def preprocess_img_to_tensor(img, img_size=None):
    """
    Preprocesses an image numpy array into a normalized tensor before
    presenting it to a Pytorch model. Adjusts the dimensions if necessary.
    """
    # This normalization only applies to the imagenet dataset we have.
    transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
        mean=[-0.01618503, -0.01468056, -0.01345447],
        std=[0.09958131, 0.0980152,  0.10090139],),
    ])
    if img.shape.index(3) == 0:
        img = np.transpose(img, (1, 2, 0))
    img_tensor = transform(img).type('torch.FloatTensor')
    
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    if img_size is not None:
        resize = T.Resize(img_size)
        img_tensor = resize(img_tensor)

    device = ('mps' if torch.has_mps else 'cpu')
    return img_tensor.to(device)


def preprocess_img_for_plot(img, norm=True):
    """
    Normalizes an image and makes sure it is in (height, width, 3) format.
    """
    if norm:
        img = img - img.min()
        img = img/img.max()
    if (len(img.shape)== 2):
        return img
    if (len(img.shape) == 4):
        img = np.squeeze(img)
    if (img.shape.index(3) == 0):
        img = np.transpose(img,(1,2,0))
    return img


def make_box(box_indicies):
    """
    Given box indicies in (vx_min, hx_min, vx_max, hx_max) format, returns a
    matplotlib.patches.Rectangle object. Example usage:

        plt.imshow(img)
        ax = plt.gca()
        rect = make_box((0, 0, 100, 50))
        ax.add_patch(rect)

    This script plots a red rectangle box with height 100 and width 50 on the
    top-left corner of the img.
    """
    vx_min, hx_min, vx_max, hx_max = box_indicies
    top_left = (hx_min, vx_min)  # (x, y) format.
    height = vx_max - vx_min + 1
    width = hx_max - hx_min + 1
    rect = patches.Rectangle(top_left, width, height, linewidth=1,
                             edgecolor='r', facecolor='none')
    return rect


def tensor_to_img(img_tensor):
    """
    Converts img_tensor into a numpy array that can be plotted by plt.imshow().
    """
    img = img_tensor.clone().detach()

    if len(img.shape) == 3 and img.shape[1] == 3:
        # If the image has RGB channels.
        return np.transpose(torch.squeeze(img),(1,2,0))
    else:
        # Plot only the first channel if there are more than three of them.
        return img[0,0,...].numpy()


class ImgDataset(Dataset):
    """
    A Dataset object of image dataset located in a directory. The generator
    returns torch.tensor that has been normalized to be roughly between
    [-1, 1]. Note that the labels of image are all set to zero because this
    project does not care about the labels. Example usage:

        num_image = 100
        img_dir = Path(__file__).parent.parent.parent.joinpath('data/imagenet')
        img_names = [f"{i}.npy" for i in range(num_images)]
        imagenet_data = ImgDataset(img_dir, img_names)
        for img, label in imagenet_data:
            outputs = model(img)

    This fatches the first 100 images from the imagenet folder and presents
    them to the model.
    """
    def __init__(self, img_dir, img_names):
        """
        Contructs an ImgDataset object.
        
        Parameters
        ----------
        img_dir : str or path-like
            The directory of the images.
        img_names : list of strs or pth-likes
            The names of the image files.
        """
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = preprocess_img_to_tensor

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.load(img_path)
        img_tensor = self.transform(img)
        label = 0  # Don't care about labels.

        return img_tensor, label


if __name__ == "__main__":
    """
    Testing ImgDataset: plot the first <num_images> images and plot the image
    no.<image_idx>.
    """
    num_images = 100
    image_idx = 0
    
    img_dir = Path(__file__).parent.parent.parent.parent.joinpath('data/imagenet')
    img_names = [f"{i}.npy" for i in range(num_images)]
    imagenet_data = ImgDataset(img_dir, img_names)
    imagenet_dataloader = DataLoader(imagenet_data, batch_size=64, shuffle=False)
    imgs, _ = next(iter(imagenet_dataloader))
    img = preprocess_img_for_plot(imgs[image_idx])
    plt.imshow(img)
    plt.show()


def one_sided_zero_pad(patch, desired_size, box):
    # Return original patch if it is the right size.
    if patch.shape == desired_size:
        return patch

    vx_min, hx_min, vx_max, hx_max = box
    touching_top_edge = (vx_min == 0)
    touching_left_edge = (hx_min == 0)

    padded_patch = np.zeros((desired_size[0], desired_size[1]))
    patch_h, patch_w = patch.shape

    if touching_top_edge and touching_top_edge:
        padded_patch[-patch_h:, -patch_w:] = patch
    elif touching_top_edge:
        padded_patch[-patch_h:, :patch_w] = patch
    elif touching_left_edge:
        padded_patch[:patch_h, -patch_w:] = patch
    else:
        padded_patch[:patch_h, :patch_w] = patch

    return padded_patch


if __name__ == "__main__":
    patch_h, patch_w = (20, 15)
    patch = np.ones((patch_h, patch_w))
    desired_size = (50, 40)
    
    vx_min = 0
    hx_min = 0
    box = (vx_min, hx_min, vx_min + patch_h, hx_min + patch_w)
    padded_patch = one_sided_zero_pad(patch, desired_size, box)
    print(box)
    plt.imshow(padded_patch)
