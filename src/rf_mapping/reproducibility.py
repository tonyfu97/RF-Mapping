import sys
import random

import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm

sys.path.append('../..')
import src.rf_mapping.constants as c

__all__ = ['set_seeds']

def set_seeds(seed=123):
    """
    Sets seeds for randomly initialized models. Should be called everytime
    an instance of the model is created.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Importing modules needed for tests below.
    from src.rf_mapping.hook import ConvUnitCounter
    from src.rf_mapping.net import get_truncated_model
    
    def truncated_model_test(model1, model2):
        counter = ConvUnitCounter(model1)
        layer_indices, _ = counter.count()

        for layer_idx in tqdm(layer_indices):
            tm1 = get_truncated_model(model1, layer_idx)
            tm2 = get_truncated_model(model2, layer_idx)
            y1 = tm1(torch.ones((1, 3, 227, 227)))
            y2 = tm2(torch.ones((1, 3, 227, 227)))
            assert(torch.allclose(y1, y2))
        print(f"All truncated model tests passed.")


if __name__ == "__main__":
    print("Testing Alexnet...")
    set_seeds()
    model1 = models.alexnet().to(c.DEVICE)
    set_seeds()
    model2 = models.alexnet().to(c.DEVICE)
    
    counter = ConvUnitCounter(model1)
    layer_indices, _ = counter.count()

    for conv_i, layer_idx in enumerate(layer_indices):
        model1_conv_weights = model1.features[0].weight
        model2_conv_weights = model2.features[0].weight
        assert(torch.allclose(model1_conv_weights, model2_conv_weights))
    print(f"All untruncated model tests passed.")

    truncated_model_test(model1, model2)


if __name__ == "__main__":
    print("Testing Vgg16...")
    set_seeds()
    model1 = models.vgg16().to(c.DEVICE)
    set_seeds()
    model2 = models.vgg16().to(c.DEVICE)
    
    truncated_model_test(model1, model2)


if __name__ == "__main__":
    print("Testing Resnet18...")
    set_seeds()
    model1 = models.resnet18().to(c.DEVICE)
    set_seeds()
    model2 = models.resnet18().to(c.DEVICE)
    
    truncated_model_test(model1, model2)
