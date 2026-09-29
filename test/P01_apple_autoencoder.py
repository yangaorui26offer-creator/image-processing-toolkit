import matplotlib.pyplot as plt
from PIL import Image  # 图像加载工具
import torchvision.transforms as transforms  # 用于图像预处理
import torch.nn as nn
import torch

# 1 加载图片
img_path = '//Users//orion//尚硅谷//智图寻宝//Apple.jpg'
img = Image.open(img_path)  # 打开图片，此时图片格式为img_file格式文件

# 2 定义并应用转换操作
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])
img_tensor = transform(img)
# 此时这个像素的范围已经是(0,1)之间了，totensor()方法自动转换
# print(img_tensor.shape)

# 3 将张量转换为numpy数组
img_ndarray = img_tensor.numpy().transpose((1, 2, 0))

# 4 用matplotlib显示图片
# plt.imshow(img_ndarray)
# plt.axis('off')
# plt.show()

# 5 创建模型,自定义类
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            # 第一层卷积池化
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # 第二层卷积池化
            nn.Conv2d(16,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,3,2,1,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        return x

model = Autoencoder()

# 5 训练模型
# 定义损失函数和优化器，相当于一个回归问题
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 定义训练的轮次
n_epochs = 500
# 训练
for epoch in range(n_epochs):
    output = model(img_tensor)
    loss = loss_function(output, img_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(loss.item())

# 6 推理过程
with torch.no_grad():
    img_recon = model(img_tensor)

img_ndarray = img_recon.numpy().transpose((1, 2, 0))
plt.imshow(img_ndarray)
plt.axis('off')
plt.show()
