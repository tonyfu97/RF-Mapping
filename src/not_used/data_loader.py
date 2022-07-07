import torchvision
import torch
from torchvision.transforms import transforms


class NaturalImageDataLoader:
    def __init__(self, path, batch_size=64):
        self.device = torch.device("mps" if torch.has_mps else "cpu")

        