#!/usr/bin/env python3
import torch
import torch.nn as nn
from base_module import UNetDownBlock, UNetDownBlock1, UNetUpBlock, UNetUpBlock1, conv3x3, UNetConvBlock

class UNet(nn.Module):

    def __init__(self, in_chn=3, wf=32, depth=5, relu_slope=0.2):
        super(UNet, self).__init__()
        self.depth = depth
        
        self.encoder1 = UNetDownBlock1(in_chn, (2 ** 0) * wf, relu_slope) # 64
        # 128
        self.encoder = nn.Sequential(
            UNetDownBlock((2 ** 0) * wf, (2 ** 1) * wf, relu_slope),  # 32
            UNetDownBlock((2 ** 1) * wf, (2 ** 2) * wf, relu_slope),  # 16
            UNetDownBlock((2 ** 2) * wf, (2 ** 3) * wf, relu_slope),  # 8
        )

        self.conv1 = UNetConvBlock((2 ** 3) * wf, (2 ** 4) * wf, relu_slope)
        self.decoder = nn.Sequential(
            UNetUpBlock((2 ** 4) * wf, (2 ** 3) * wf, relu_slope),  # 16
            UNetUpBlock((2 ** 3) * wf, (2 ** 2) * wf, relu_slope),  # 32
            UNetUpBlock((2 ** 2) * wf, (2 ** 1) * wf, relu_slope),  # 64
        )
        
        self.decoder1 = UNetUpBlock1((2 ** 1) * wf, (2 ** 0) * wf, relu_slope)  # 128
        self.conv2 = conv3x3((2 ** 0) * wf, 3, bias=True)

    def forward(self, x, c):
        # encoder stage
        x = self.encoder1(x, c)
        x = self.encoder(x)
        x = self.conv1(x)
        # decoder stage
        x = self.decoder(x)
        x = self.decoder1(x, c)
        res = self.conv2(x)
        return res



