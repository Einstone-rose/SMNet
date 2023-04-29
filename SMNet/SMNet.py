import math
import torch
import torch.nn as nn
from base_module import*
from u_net import UNet


class SMNet(nn.Module):
    def __init__(self, in_channel=3):
        super(SMNet, self).__init__()
        self.smn = UNet(in_channel)
        self.CondNet = nn.Sequential(
            nn.Conv2d(1, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

    def forward(self, x, Map):
        c = self.CondNet(Map)
        res = self.smn(x, c)
        return res
