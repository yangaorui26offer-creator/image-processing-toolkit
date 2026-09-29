# 数据预处理路径
# 常量一般均大写
IMG_PATH = '../common/dataset/'
IMG_HEIGHT = 68
IMG_WIDTH = 68

# 随机性与数据集划分
SEED = 42  # 随机数种子
TRAIN_RATIO = 0.75  # 训练集划分比例
TEST_RATIO = 1 - TRAIN_RATIO
NOISE_FACTOR = 0.5  # 噪声因子

# 训练相关的超参数
LEARNING_RATE = 0.001
EPOCHS = 20
TRAIN_BATCH_SIZE = 32  # mini_batch大小，正常还有test_batch_size
TEST_BATCH_SIZE = 32

# 模块名称和保存模型参数的文件名
PACKAGE_NAME = 'image_denoising'
DENOISER_MODEL_NAME = 'denoiser.pt'
