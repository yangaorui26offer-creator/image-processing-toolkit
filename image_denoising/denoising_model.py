__all__ = ['ConvDenoiser']

import torch
import torch.nn as nn

# 自定义神经网络结构类
class ConvDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 解码器
        self.t_conv1 = nn.ConvTranspose2d(8,8,kernel_size=3,stride=2,padding=0,output_padding=0)
        self.t_conv2 = nn.ConvTranspose2d(8,16,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.t_conv3 = nn.ConvTranspose2d(16,32,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.conv_out = nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        # 编码
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)
        # 解码
        x = self.t_conv1(x)
        x = torch.relu(x)
        x = self.t_conv2(x)
        x = torch.relu(x)
        x = self.t_conv3(x)
        x = torch.relu(x)
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        return x