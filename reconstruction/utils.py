import torchvision
import numpy as np
import torch
import torch.nn as nn


def tensor_to_PILimage(x):
    unloader = torchvision.transforms.ToPILImage()
    x = unloader(x)
    x = np.asarray(x)
    return x


