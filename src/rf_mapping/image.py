"""
Code to for image processing, etc.

Tony Fu, Jun 25, 2022
"""
import numpy as np
import torch
import torchvision.transforms as T


def clip(x, x_min, x_max):
    """Limits x to be x_min <= x <= x_max."""
    x = min(x_max, x)
    x = max(x_min, x)
    return x


def normalize_image(image):
    """Normalize pixel values to be between [-1, 1]."""
    norm_image = image - image.min()
    norm_image = norm_image/norm_image.max()
    return norm_image


def preprocess_image(image, image_size=None):
    """Preprocess image before presenting it to a Pytorch model."""
    norm_image = normalize_image(image)
    norm_image = np.expand_dims(norm_image, axis=0)
    image_tensor = torch.from_numpy(norm_image).type('torch.FloatTensor')

    if image_size is not None:
        resize = T.Resize(image_size)
        image_tensor = resize(image_tensor)

    return image_tensor


def tensor_to_image(image_tensor):
    image = image_tensor.clone().detach()

    if len(image.shape) == 3 and image.shape[1] == 3:
        return np.transpose(torch.squeeze(image),(1,2,0))
    else:
        return image[0,0,...].numpy()
