# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:17:12 2020

@author: sergv
"""

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,skip_step=3):
        super(Net, self).__init__()
        
        self.skip_step = skip_step
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 3)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 64, 3)
        self.conv9_bn = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, 3)
        self.conv10_bn = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, 3)
        self.conv11_bn = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, 3)
        self.conv12_bn = nn.BatchNorm2d(64)
        self.conv13 = nn.Conv2d(64, 64, 3)
        self.conv13_bn = nn.BatchNorm2d(64)
        self.conv14 = nn.Conv2d(64, 64, 3)
        self.conv14_bn = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 64, 3)
        self.conv15_bn = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, 3)
        self.conv16_bn = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 1, 3)
        
    def forward(self, x):
		
        blk = int(self.skip_step)-1
        
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2_bn(self.conv2(x1)))
        x3 = F.leaky_relu(self.conv3_bn(self.conv3(x2)))
        x4 = F.leaky_relu(self.conv4_bn(self.conv4(x3+x1[:,:,blk:-blk,blk:-blk])))
        x5 = F.leaky_relu(self.conv5_bn(self.conv5(x4)))
        x6 = F.leaky_relu(self.conv6_bn(self.conv6(x5)))
        x7 = F.leaky_relu(self.conv7_bn(self.conv7(x6+x4[:,:,blk:-blk,blk:-blk])))
        x8 = F.leaky_relu(self.conv8_bn(self.conv8(x7)))
        x9 = F.leaky_relu(self.conv9_bn(self.conv9(x8)))
        x10 = F.leaky_relu(self.conv10_bn(self.conv10(x9+x7[:,:,blk:-blk,blk:-blk])))
        x11 = F.leaky_relu(self.conv11_bn(self.conv11(x10)))
        x12 = F.leaky_relu(self.conv12_bn(self.conv12(x11)))
        x13 = F.leaky_relu(self.conv13_bn(self.conv13(x12+x10[:,:,blk:-blk,blk:-blk])))
        x14 = F.leaky_relu(self.conv14_bn(self.conv14(x13)))
        x15 = F.leaky_relu(self.conv15_bn(self.conv15(x14)))
        x16 = F.leaky_relu(self.conv16_bn(self.conv16(x15+x13[:,:,blk:-blk,blk:-blk])))
        x17 = self.conv17(x16)
        out_sin = torch.sin(x17)
        out_cos = torch.cos(x17)
        out = torch.atan2(out_sin,out_cos)
        return out

