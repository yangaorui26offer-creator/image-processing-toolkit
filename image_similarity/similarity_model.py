__all__ = ['ConvEncoder', 'ConvDecoder']

import torch
import torch.nn as nn

# 定义编码器
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积池化
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 第二层卷积池化
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 第三层卷积池化
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 第四层卷积池化
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 第五层卷积池化
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        return x

# 定义解码器
class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层转置卷积
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        # 第二层转置卷积
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        # 第三层转置卷积
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        # 第四层转置卷积
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        # 第五层装置卷积
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2, padding=0)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.deconv5(x)
        x = self.relu5(x)
        return x

if __name__ == '__main__':
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    input_tensor = torch.randn(1, 3, 64, 64)
    encoded_tensor = encoder(input_tensor)
    decoded_tensor = decoder(encoded_tensor)
    print(decoded_tensor.shape)