import torch

__all__ = ['train_step','test_step','create_embedding']

# 定义一个轮次的训练步骤
def train_step(encoder,decoder,train_loader,loss_fn,optimizer,device):
    # 设置为训练模式
    encoder.train()
    decoder.train()
    # 累计损失值
    total_loss = 0.0
    # 遍历DataLoader，按批次训练模型
    for train_images, train_labels in train_loader:
        # 0. 将数据移动到设备上
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        # 1. 前向传播
        encode_output = encoder(train_images)
        outputs = decoder(encode_output)
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


def test_step(encoder, decoder, test_loader, loss_fn, device):
    # 设置成测试模式
    encoder.eval()
    decoder.eval()
    # 定义总测试误差
    total_loss = 0.0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            encode_output = encoder(test_images)
            outputs = decoder(encode_output)
            loss_value = loss_fn(outputs, test_labels)
            total_loss += loss_value.item()
        return total_loss / len(test_loader)


def create_embedding(encoder, full_loader, device):
    '''
    为整个数据集生成嵌入表示
    :param encoder: 训练好的编码器
    :param full_loader: 完整数据集的加载器
    :param embedding_dim: 期望嵌入的维度
    :param device: 设备
    :return: 返回嵌入张量，形状[N,C,H,W] - [N,256,2,2]
    '''
    encoder.eval()
    # 定义嵌入张量，初始为空，后续通过torch.cat做张量拼接
    embeddings = torch.empty(0)  # 初始化分配张量存储空间
    with torch.no_grad():
        for train_img,target_img in full_loader:
            train_img = train_img.to(device)
            # 前向传播,只做编码，提取特征，后续保存在cpu上后续做numpy转换
            encoded_img = encoder(train_img).cpu()
            # 将这一批次的特征结果拼接到嵌入张量中
            embeddings = torch.cat((embeddings, encoded_img), dim=0)  # 指明维度，按照第一个维度即N那个维度进行拼接叠加
        return embeddings
