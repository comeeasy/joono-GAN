import torch
import torch.nn as nn

import sys

sys.path.append("/home/joono/MinLab/models")
from Blocks import ConvBlock, ResBlock, ResTransposeBlock, TransposeConvBlock


class MNISTDiscriminator(nn.Module):
    def __init__(self, n_classes, z_dim, hidden_c):
        super().__init__()
        # B x 1 x 28 x 28 -> B x 64 x 28 x 28
        self.input_layer = ConvBlock(in_channel=1, out_channel=hidden_c, ksize=5, stride=1)
        
        self.hidden_layers = nn.Sequential(
            # B x 64 x 28 x 28 -> B x 128 x 14 x 14
            ConvBlock(in_channel=1 * hidden_c, out_channel=2 * hidden_c, ksize=3, stride=1, downsample=True),
            # B x 128 x 14 x 14 -> B x 256 x 14 x 14 
            ConvBlock(in_channel=2 * hidden_c, out_channel=4 * hidden_c, ksize=3, stride=1),
            # B x 256 x 14 x 14 -> B x 512 x 7 x 7
            ConvBlock(in_channel=4 * hidden_c, out_channel=8 * hidden_c, ksize=3, stride=1, downsample=True),
        )
        self.output_layer = nn.Sequential(
            # B x 512 x 7 x 7 -> B x 100 x 7 x 7
            ConvBlock(in_channel=8 * hidden_c, out_channel=z_dim + n_classes, ksize=1, stride=1),
        )
        self.z_layer = nn.Sequential(
            # B x 100 x 7 x 7 -> B x 100 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # B x 100 x 1 x 1 -> B x 1
            nn.Linear(z_dim + n_classes, 1),
            nn.Sigmoid()
        )

        self.c_layer = nn.Sequential(
            # B x 100 -> B x 10 x 7 x 7
            nn.Conv2d(z_dim + n_classes, n_classes, 1, 1),
            nn.BatchNorm2d(10),
            # B x 10 x 7 x 7 -> B x 10 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        z = self.z_layer(out)
        c = self.c_layer(out)

        return z, c


class Generator(nn.Module):
    def __init__(self, n_classes, z_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            # B x 100 x 1 x 1 -> B x 100 x 7 x 7 
            ResTransposeBlock(z_dim + n_classes, z_dim + n_classes, ksize=7, stride=1),
            # B x 100 x 7 x 7 -> B x 512 x 7 x 7
            ResTransposeBlock(z_dim + n_classes, 8 * hidden_dim, upsample=False),
        )
        self.hidden_layers = nn.Sequential(
            # B x 512 x 7 x 7 -> B x 256 x 14 x 14
            ResTransposeBlock(8 * hidden_dim, 4 * hidden_dim, upsample=True),
            # B x 256 x 14 x 14 -> B x 128 x 14 x 14
            ResTransposeBlock(4 * hidden_dim, 2 * hidden_dim, upsample=False),
            # B x 128 x 14 x 14 -> B x 64 x 28 x 28
            ResTransposeBlock(2 * hidden_dim, 1 * hidden_dim, upsample=True),
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        out = torch.cat((x, c), dim=1)

        out = self.input_layer(out)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        return out

class CIFAR10Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # B x 3 x 28 x 28 -> B x 64 x 28 x 28, 3 is for RGB image
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )
        self.hidden_layers = nn.Sequential(
            ResBlock(in_channels=32, out_channels=64),
            ResBlock(in_channels=64, out_channels=128, downsample=True),
            ResBlock(in_channels=128, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=512),
            ResBlock(in_channels=512, out_channels=512, downsample=True),
            ResBlock(in_channels=512, out_channels=1024),
            ResBlock(in_channels=1024, out_channels=1024, downsample=True),
        )
        self.output_layer = nn.Sequential(
            ResBlock(in_channels=1024, out_channels=n_classes),
        )
        self.z_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_classes, 1),
            nn.Sigmoid()
        )

        self.c_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_classes, n_classes, bias=True)
        )
        

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        z = self.z_layer(out)
        c = self.c_layer(out)

        return z, c


class CIFAR10Generator(nn.Module):
    def __init__(self, n_classes, z_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            ResTransposeBlock(z_dim + n_classes, 1024, ksize=7, stride=1),
            ResTransposeBlock(1024, 1024, upsample=False),
        )
        self.hidden_layers = nn.Sequential(
            ResTransposeBlock(1024, 512, upsample=True),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 128),
            ResBlock(128, 128),
            ResTransposeBlock(128, 128, upsample=True),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1), # 3 is for RGB image 
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x, c):
        out = torch.cat((x, c), dim=1)

        out = self.input_layer(out)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        return out