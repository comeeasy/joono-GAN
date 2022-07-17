import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ksize: int, stride: int, downsample=False):
        super().__init__()

        self.pad = ksize // 2
        self.downsample = downsample
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel, kernel_size=ksize, 
                stride=stride, padding=self.pad),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d((2, 2))

        self._init_weights(self.layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.xavier_normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


    def forward(self, x):
        out = self.layers(x)

        if self.downsample:
            out = self.pool(out)

        return out

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=False, ksize=None, stride=None):
        super().__init__()

        self.upsample = upsample
        self.stride = 2 if upsample else 1
        self.ksize = 4 if upsample else 3
        self.pad = 1

        if ksize:
            self.ksize = ksize
            self.pad = 0
        if stride:
            self.stride = stride

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channel, out_channels=out_channel,
                kernel_size=self.ksize, stride=self.stride, padding=self.pad
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._init_weights(self.layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.xavier_normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


    def forward(self, x):
        out = self.layers(x)

        return out