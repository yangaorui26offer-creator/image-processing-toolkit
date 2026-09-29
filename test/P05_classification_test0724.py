import torch
import torch.nn as nn
import PIL.Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.datasets import images
from sympy.codegen.ast import Raise
from torch.utils.data import Dataset, DataLoader,random_split
import re
import os

# 构造正则函数，将图片名称按照字母及数字大小排序
# 1.传入的参数为图片名称列表
def image_names_sorted(images_names):
    # 1.1 构造lambda函数做convert转换,如果该文本为数字则转化成为int类型，否则转换为小写字母
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 1.2 构造lambda正则函数，将字母数字切分，并存储在list列表中，且遍历该list，进行字母数字类型转换
    alphanum_key = lambda image: [convert(x) for x in re.split('([0-9]+)', image)]
    return sorted(images_names, key=alphanum_key)  # 默认按照大小升序排序，通过sort函数会生成一个新的列表

# 2. 构造Dataset数据集
class ImageDataset(Dataset):
    # 2.1 传入的参数为文件路径以及transforms
    def __init__(self,images_path,labels_path,transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        # 2.2 读取图片，并将图片名称存储在list中
        self.images_names = image_names_sorted(os.listdir(self.images_path))
        # 2.3 读取label文件，并将其转换为字典，方便进行后续调用
        self.labels = pd.read_csv(self.labels_path)
        self.labels.dict = dict(zip(self.labels['id'], self.labels['target']))  # 定义拉链函数，将图片的id与标签值一一对应

    def __len__(self):
        return len(self.images_names)  # 返回图片名字list的长度

    def __getitem__(self, index):
        # 2.4 读取图片，并进行图片的剪裁与转换
        img = PIL.Image.open(self.images_path + self.images_names[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        else:
            ValueError('Transform not defined')
        x = img
        y = torch.tensor(self.labels.dict[index])
        return x, y

# 3. 进行测试
if __name__ == '__main__':
    # 定义transforms
    transform = transforms.Compose([
        transforms.Resize((68,68)),
        transforms.ToTensor(),
    ])
    # 对Dataset数据进行训练集和测试集的划分
    full_dataset = ImageDataset(images_path='../common/dataset/',labels_path='../common/fashion-labels.csv',transform=transform)
    train_dataset,test_dataset = random_split(full_dataset,[0.75,0.25])
    # 装在train_dataloader以及test_dataloader
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,drop_last=True)
    # 任取一组x,y进行形状测试
    x,y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)



