# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:43:35 2019

@author: Kaifan JI
"""
import torch.nn as nn



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)    

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv3 = conv3x3(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
       
    def forward(self, out):
        residual = out
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn(out)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, channel_in, channel_conv, layers, channel_out):
        super(ResNet, self).__init__()
        self.in_channels = channel_in
        self.bn0 = nn.BatchNorm2d(channel_in) 
        self.conv =  conv3x3(channel_in, channel_conv) 
        self.bn = nn.BatchNorm2d(channel_conv)
        self.relu = nn.ReLU(inplace=True)
        self.block_layer = self.make_layer(ResidualBlock, channel_conv, layers)
        self.last_conv =  conv3x3(channel_conv, channel_out)
      
    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, out):
        out = self.bn0(out) 
        out = self.conv(out) 
        out = self.bn(out) 
        out = self.relu(out) 
        out = self.block_layer(out)
        out = self.last_conv(out)
        return out
