import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import functional as F
from torch import nn
import numpy as np

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
