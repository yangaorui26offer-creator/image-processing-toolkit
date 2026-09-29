import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,random_split

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条工具

# 导入自定义组件
from common import utils
from classification_config import *
from classification_data import *
from classification_model import *
from classification_engine import *

if __name__=='__main__':
    # 0. 准备工作
    # 检测GPU是否可用并定义设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 指定随机数种子，去除训练的不确定性
    utils.seed_everything(SEED)

    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    # 1. 创建数据集
    print('---1.创建数据集---')
    dataset = ImageLabelDataset(IMG_PATH,LABELS_PATH,transform)
    # 划分训练集和测试集
    train_dataset,test_dataset = random_split(dataset,[TRAIN_RATIO,TEST_RATIO])
    print('---创建数据集完成')

    # 2. 创建数据加载器
    print('---2.创建数据加载器---')
    train_loader = DataLoader(
        train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        drop_last = True,
    )
    test_loader = DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE,drop_last=True)
    print('---创建数据加载器完成')

    # 3. 训练模型
    # 定义模型，损失函数和优化器
    classifier = Classifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(),lr=LEARNING_RATE)

    classifier.to(device)
    min_test_loss = 9999  # 初始化最小测试误差为极大值

    print('---3.开始训练模型---')

    for epoch in tqdm(range(EPOCHS)):
        # 训练一轮
        train_loss= train_step(classifier,train_loader,loss_fn,optimizer,device)
        print(f'\nepoch:{epoch+1},train_loss:{train_loss}')

        # 测试一轮，获取测试误差
        test_loss, test_correct_num = test_step(classifier,test_loader,loss_fn,device)
        accuracy = test_correct_num / len(test_dataset)
        print(f'\nepoch:{epoch+1},test_loss:{test_loss},Accuracy:{accuracy:.6f}')

        # 判断当前测试误差是否小于历史最小值，如果小于则保存模型参数
        if test_loss < min_test_loss:
            print('Saving model')
            min_test_loss = test_loss
            torch.save(classifier.state_dict(),CLASSIFIER_MODEL_NAME)
        else:
            print('Not saving model')

    print('---训练完成---')