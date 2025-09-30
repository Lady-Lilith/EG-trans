import numpy as np
import torch
from torch import nn
from torch.nn import init
from param import parameter_parser

class ParNetAttention(nn.Module):

    def __init__(self, channel=256,dropout=0.2):
        super().__init__()
        args = parameter_parser()
        self.args = args
        self.dropout = nn.Dropout(dropout)
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),

        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)

        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.silu = nn.SiLU()
        self.mlp = nn.Sequential(
            nn.Linear(self.args.circRNA_number+self.args.disease_number, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.args.disease_number),

        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.unsqueeze(-1)
        x = x.permute(0,2,3,1)
        # x = x.permute(0,1,2,3)
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x

        # y = self.silu(x1 + x2 + x3)
        y = x1+x2+x3
        y = y.permute(0,3,2,1)
        y = self.mlp(y)
        y = y.squeeze(0)
        # y = y.permute(2,0,1)
        y = y.permute(0,2,1)
        y = y.squeeze(-1)
        return y