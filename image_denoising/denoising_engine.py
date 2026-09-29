__all__ = ['train_step', 'test_step']

import torch

def train_step(denoiser,train_loader,loss_fn,optimizer,device):
    '''
    执行一轮epoch的完整训练步骤
    :param denoiser: 降噪器模型
    :param train_loader: 训练数据加载器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 返回当前轮次的平均训练损失
    '''
    # 设置为训练模式
    denoiser.train()
    # 累计损失值
    total_loss = 0.0
    # 遍历DataLoader，按批次训练模型
    for train_images, train_labels in train_loader:
        # 0. 将数据移动到设备上
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        # 1. 前向传播
        outputs = denoiser(train_images)
        # 2. 计算损失
        loss_value = loss_fn(outputs, train_labels)
        # 3. 反向传播
        loss_value.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()
        # 6. 累加损失值
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def test_step(denoiser,test_loader,loss_fn,device):
    # 设置成测试模式
    denoiser.eval()
    # 定义总测试误差
    total_loss = 0.0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = denoiser(test_images)
            loss_value = loss_fn(outputs, test_labels)
            total_loss += loss_value.item()
        return total_loss / len(test_loader)


