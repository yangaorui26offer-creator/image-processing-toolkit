__all__ = ['train_step','test_step']

import torch

def train_step(classifier,train_loader,loss_fn,optimizer,device):
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)  # 前向传播
        loss_value = loss_fn(outputs,labels)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def test_step(classifier,test_loader,loss_fn,device):
    total_loss = 0
    correct_num = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            loss_value = loss_fn(outputs,labels)
            total_loss += loss_value.item()
            # 预测分类标签并计算正确个数
            pred = outputs.argmax(dim=1)
            correct_num += (pred == labels).sum().item()
        return total_loss / len(test_loader), correct_num
