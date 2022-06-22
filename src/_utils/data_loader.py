import torchvision
import torch
from torchvision.transforms import transforms


class NaturalImageDataLoader:
    def __init__(self, path, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        