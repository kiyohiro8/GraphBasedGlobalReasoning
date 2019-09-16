# -*- coding: utf-8 -*-

from distutils.util import strtobool

import torch
from torch import nn

from layers import ResNet50, FCNHead

class FCNwithGloRe(nn.Module):
    def __init__(self, params):
        super(FCNwithGloRe, self).__init__()
        common_params = params["common"]
        network_params = params["network"]

        num_class = common_params["num_class"]
        image_size = common_params["image_size"]
        
        use_glore = network_params["use_glore"]
        base_channels = network_params["base_channels"]
        multi_grid = network_params["multi_grid"]
                
        self.resnet = ResNet50(base_channels, multi_grid)
        base_channels *= 16
        self.head = FCNHead(base_channels, image_size, num_class, use_glore)
        
    def forward(self, x):
        image_size = x.size()[2:]
        x = self.resnet(x)
        out = self.head(x, image_size)
        return out

class UNetwithGloRe(nn.Module):
    def __init__(self):
        super(UNetwithGloRe, self).__init__()
