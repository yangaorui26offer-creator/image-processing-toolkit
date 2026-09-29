__all__ = ['ImageLabelDataset']

import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,random_split  # 数据集处理模块

from PIL import Image
import re

from image_classification.classification_config import *



# 1.数据提取
# 1.1 先构造一个函数，能够将图片文件名按照字母及数字大小排序
def sorted_alphanum(image_names):  # 传入的是一个列表
    # 定义转换函数
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 定义key
    alphanum_key = lambda img_name: [convert(x) for x in re.split('([0-9]+)', img_name)]
    return sorted(image_names, key=alphanum_key)

# 1.2 自定义数据集类型，元素(image,label)
class ImageLabelDataset(Dataset):
    def __init__(self,image_dir,label_path,transform=None):
        # 将文件的读取路径和图片的转换操作transform定义为属性，方便后面随时调用
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))  # 获取目录下所有图片文件名，并按照字母数字顺序混合排序
        self.labels = pd.read_csv(label_path)
        # 用拉链函数创造一个字典，便于后续查询target，拉链返回的格式为{id:label}
        self.label_dict = dict(zip(self.labels['id'],self.labels['target']))

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
        # 4 在字典中找出图片对应的标签
        label = torch.tensor(self.label_dict[idx])
        return  tensor_image,label

# 测试创建数据集
if __name__ == '__main__':
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_HEIGHT,IMG_WIDTH)),
        T.ToTensor(),
    ])
    dataset = ImageLabelDataset(IMG_PATH,LABELS_PATH,transform=transform)
    print(len(dataset))