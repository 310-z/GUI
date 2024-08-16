import torch
from torch import nn


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        # 对空间信息进行压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 取出batch size和通道数
        b, c, _, _ = x.size()
        # b,c,w,h -> b,c,1,1 -> b,c 压缩与通道信息学习
        y = self.avg_pool(x).view(b, c)
        # b,c->b,c->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        # 激励操作
        return x * y.expand_as(x)




