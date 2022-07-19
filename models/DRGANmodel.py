import torch
import torch.nn as nn
import torchvision


class DRGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
