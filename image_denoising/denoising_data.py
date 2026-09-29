__all__ = ['ImageDataset']  # 暴露给外部，需要的时候直接调用

import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
from denoising_config import *

# 自定义函数，对图片名按照字母数字混合排序
def sorted_alphanum(img_names):
    # 转换函数:将数字部分转化为int，将字符串转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name:[convert(x) for x in re.split('([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

class ImageDataset(Dataset):
    def __init__(self,image_dir,transform=None):
        # 将文件的读取路径和图片的转换操作transform定义为属性，方便后面随时调用
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))  # 获取目录下所有图片文件名，并按照字母数字顺序混合排序

    def __len__(self):
        return len(self.image_names)

    # 传入图片id，获取数据集元素(x,y)
    def __getitem__(self, idx):
        # 1 根据索引号构建图片的完整路径
        image_loc = os.path.join(self.main_dir,self.image_names[idx])
        # 2 使用PIL打开图片,并转换格式，变成RGB三通道数据
        image = Image.open(image_loc).convert('RGB')
        # 3 利用transform转换成tensor
        if self.transform is not None:
            tensor_image = self.transform(image)
        else:
            # 如果为None，抛出异常
            raise ValueError('transform cannot be None!')
        # 4 向原始图像中增加随机噪声
        noise_img = tensor_image + NOISE_FACTOR * torch.randn_like(tensor_image)
        noise_img = torch.clamp(noise_img, 0., 1.)  # 由于transform后的图像像素范围都在0到1之间，故此时需要限制像素范围还在0，1之间
        # 返回(噪声图片，原始图片)
        return noise_img, tensor_image

# 测试流程
if __name__ == '__main__':
    image_names = os.listdir(IMG_PATH)
    # print(sorted_alphanum(image_names))