import PIL
import time, json
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn
from utils import device
import random

class CNNNet(nn.Module):
    def __init__(self, params):
        super(CNNNet, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        num_classes = data_params.get("num_classes", 16)
        self.patch_size = patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        self.wh = self.patch_size * self.patch_size

        conv2d_out = 64
        kernal = 3
        padding = 1
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal), stride=1, padding=(padding,padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            # featuremap 
            nn.Conv2d(in_channels=conv2d_out,out_channels=conv2d_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU()
        )

        self.projector = nn.Linear(self.wh, 1) 

        dim = conv2d_out
        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )

    def encoder_block(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height
        '''
        x_pixel = x 

        x_pixel = self.conv2d_features(x_pixel)
        b, s, w, h = x_pixel.shape
        img = w * h

        x_pixel = rearrange(x_pixel, 'b s w h-> b s (w h)') # (batch, spe, w*h)
        x_pixel = self.projector(x_pixel) # (batch, spe, 1)
        x_pixel = rearrange(x_pixel, 'b s 1 -> b s')
        # print(x_pixel.shape)

        return self.classifier_mlp(x_pixel)

    def forward(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x = self.encoder_block(x)
        return  logit_x