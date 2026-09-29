__all__ = ['Classifier']

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1)
        # 通用池化层,池化层没有参数更新，故可以定义通用池化层
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 全连接层
        self.linear = nn.Linear(16*16*16,n_classes)

    def forward(self, x):
        # 第一层卷积
        x = torch.relu(self.conv1(x))
        # 第一层池化
        x = self.pool(x)
        # 第二层卷积
        x = torch.relu(self.conv2(x))
        # 第二层池化
        x = self.pool(x)
        # 扁平化处理
        x = x.reshape(x.shape[0],-1)
        # 线性层
        x = self.linear(x)
        return x

# 测试模型
if __name__ == '__main__':
    x = torch.randn(1,3,64,64)  # 输入数据
    model = Classifier()
    output = model(x)
    print(output.shape)

