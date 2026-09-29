import numpy as np
import torch
import random
import os


def seed_everything(seed):
    # 1. Python内置随机库、哈希随机
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Numpy随机种子
    np.random.seed(seed)

    # 3. PyTorch CPU全局种子
    torch.manual_seed(seed)

    # 4. Mac MPS GPU专属种子（加判断避免CPU环境报错）
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 5. 卷积算子确定性配置（消除底层随机算法选择）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. 全局强制确定性计算（最高等级复现保障）
    torch.use_deterministic_algorithms(True)


# 调用示例，固定种子42
seed_everything(42)
