import torch
import torch.nn as nn
import os
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np  


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def dilationConv(in_chn, out_chn, dilation, bias=True):
    # padding =
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, dilation=dilation, padding=dilation, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias) # map 变为原来的两倍
    return layer


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out

class UNetDownBlock1(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetDownBlock1, self).__init__()
        self.block = UNetConvBlock(in_size, out_size, relu_slope)
        self.down_sample = conv_down(2 * out_size, out_size, bias=False)
        self.sft = SFTLayer(out_size)
        #self.sconv = nn.Conv2d(out_size, out_size, 3, 1, 1)
        self.se = SELayer(2 * out_size)

    def forward(self, x, c):
        out = self.block(x)
        fea = self.sft(out, c)
        res = self.se(torch.cat([out, fea], dim=1))
        out_down = self.down_sample(res)
        return out_down

class UNetDownBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetDownBlock, self).__init__()
        self.block = UNetConvBlock(in_size, out_size, relu_slope)
        self.down_sample = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out = self.block(x)
        out_down = self.down_sample(out)
        return out_down


class SFTLayer(nn.Module):
    def __init__(self, out_size=32):
        super(SFTLayer, self).__init__()
        self.alpha_conv0 = nn.Conv2d(32, 32, 1)
        self.alpha_conv1 = nn.Conv2d(32, out_size, 1)
        self.beta_conv0 = nn.Conv2d(32, 32, 1)
        self.beta_conv1 = nn.Conv2d(32, out_size, 1) 

        self.leaky_relu0 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leaky_relu1 = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, c):
        # x: feat, cond: c
        alpha = self.alpha_conv1(self.leaky_relu0(self.alpha_conv0(c)))
        beta = self.beta_conv1(self.leaky_relu1(self.beta_conv0(c)))
        return x * alpha + beta
         

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)  # 扩大两倍
        self.conv_block = UNetConvBlock(out_size, out_size, relu_slope)

    def forward(self, x):
        upx = self.up(x)
        out = self.conv_block(upx)# 不一定是最优的
        return out


class UNetUpBlock1(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock1, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)  # 扩大两倍
        self.conv_block = UNetConvBlock(2 * out_size, out_size, relu_slope)
        self.sft = SFTLayer(out_size)
        #self.sconv = nn.Conv2d(out_size, out_size, 3, 1, 1)
        self.se = SELayer(2 * out_size)

    def forward(self, x, c):
        upx = self.up(x)
        #c = torch.cat([upx, c], dim=1)
        fea = self.sft(upx, c)
        res = self.se(torch.cat([upx, fea], dim=1))
        out = self.conv_block(res)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        b_, c_, h_, w_ = x.size()
        x = x.view(b_, h_ * w_, c_)
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        #y = self.avg_pool(x).view(b, c)
        y = x.mean(dim=-1).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        x = x.view(b_, c_, h_, w_)
        return x

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

