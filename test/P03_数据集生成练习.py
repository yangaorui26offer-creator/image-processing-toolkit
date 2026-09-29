import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
import PIL.Image as Image  # 做数据的读取以及RBG通道的剪裁
import os  # 通过操作系统来读取加载图片集

# 1、定义一个数据集的类
class ImageDataset(Dataset):
    def __init__(self,img_path,transform=None):
        self.img_path = img_path  # 此处为读取数据集的路径
        # 读取图片集中所有的索引号，并存储在一个list中
        self.img_names = os.listdir(self.img_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):  # 此处的index为调用img_names中的name的索引
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.img_path,img_name)).convert('RGB')  # 调用了该图片,并转换成'RGB'标准格式的数据
        # 调用transform将图片转换成为(3,68,68)的形状，并转换成为tensor格式
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            raise ValueError('transform cannot be None!')
        # 创造添加噪声的图片
        noise_factor = 0.5
        noise_img = img_tensor + noise_factor * torch.randn_like(img_tensor)
        # 加上噪声的数据要注意像素点的范围限制
        noise_img = torch.clamp(noise_img, 0., 1.)
        return noise_img, img_tensor

# 2、测试流程
transform = transforms.Compose([
    transforms.Resize((68,68)),
    transforms.ToTensor(),
])

# 2.1 创造一个Dataset集
image_dataset = ImageDataset('../common/dataset/',transform=transform)

# 2.2 对数据集进行划分
train_dataset,test_dataset = random_split(image_dataset,[0.75,0.25])

# 2.3 对数据进行装载
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,drop_last=False)

for index,(x,y) in enumerate(train_loader):
    print(x.shape)
    print(y.shape)
    break