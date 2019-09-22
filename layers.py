# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn.functional import interpolate

class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2
        
        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)
        
        self.relu = nn.ReLU()
        
        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)
        
        # このunitに入力された時のチャンネル数と合わせるためのconv layer
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        
        B = self.theta(x).view(-1, self.N, L)

        phi = self.phi(x).view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)

        V = torch.bmm(B, phi) / L #著者コード中にある謎割り算
        V = self.relu(self.node_conv(V))
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2)))
        
        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)
        
        return x + y
    
    
class ResBlock(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if downsample is None:
            # 以下の条件を満たさない場合、ConvLayerによるdownsamplingを行う
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels))
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet50(nn.Module):
    def __init__(self, base_channels=64, multi_grid=False):
        
        super(ResNet50, self).__init__()
        block = ResBlock
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, base_channels//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels//2, base_channels//2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels//2, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = make_resblock(block, base_channels*1, base_channels*2, num_blocks=3)
        self.block2 = make_resblock(block, base_channels*2, base_channels*4, num_blocks=4, stride=2)
        
        
        self.block3 = make_resblock(block, base_channels*4, base_channels*8,
                                    num_blocks=6, stride=1, dilation=2)
            
        if multi_grid:
            self.block4 = make_resblock(block, base_channels*8, base_channels*16, num_blocks=3, stride=1,
                                           dilation=4, multi_grid=[4, 8, 16])

        else:
            self.block4 = make_resblock(block, base_channels*8, base_channels*16, num_blocks=3, stride=1,
                                           dilation=4)        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    
def make_resblock(block, in_channels, out_channels, num_blocks, stride=1, dilation=1, 
                  downsample=None, multi_grid=None):
    layers = []
    if multi_grid is not None:
        multi_dilations = multi_grid
    else:
        multi_dilations = [dilation] * num_blocks
    assert len(multi_dilations) == num_blocks, "multi_dilationsの要素数はブロック数と等しくなるように与えてください"
    
    if multi_grid:
        layers.append(block(in_channels, out_channels, stride, dilation=multi_dilations[0],
                            downsample=downsample, previous_dilation=dilation))
    elif dilation == 1 or dilation == 2:
        layers.append(block(in_channels, out_channels, stride, dilation=1,
                            downsample=downsample, previous_dilation=dilation))
    elif dilation == 4:
        layers.append(block(in_channels, out_channels, stride, dilation=2,
                            downsample=downsample, previous_dilation=dilation))
    else:
        raise RuntimeError("=> unknown dilation size: {}".format(dilation))

    for i in range(1, num_blocks):
        layers.append(block(out_channels, out_channels, dilation=multi_dilations[i],
                            previous_dilation=dilation))


    return nn.Sequential(*layers)

class FCNHead(nn.Module):
    def __init__(self, in_channels, image_size, num_class, use_glore=True):
        super(FCNHead, self).__init__()
        self.image_size = image_size

        inter_channels = in_channels // 4
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.use_glore = use_glore
        if self.use_glore:
            self.gcn = GloRe(inter_channels)

        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv53 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.Dropout2d(0.2),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, num_class, 3, padding=1, bias=False))

    def forward(self, x, image_size):
        x = self.conv51(x)
        if self.use_glore:
            x = self.gcn(x)
        x = self.conv52(x)
        x = interpolate(x, image_size)
        x = self.conv53(x)
        #x = x[:, :, 1:-1, 1:-1] # conv53のpaddingで拡大してしまった分を除去
        
        output = self.conv6(x)

        return output
    