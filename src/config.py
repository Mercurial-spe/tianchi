"""
配置管理模块
集中管理项目中的所有参数配置
"""

# 数据路径
TRAIN_DATA_PATH = "data/train.txt"
TEST_DATA_PATH = "data/A.txt"
TRAIN_IMG_PREFIX = "data/train/"
TEST_IMG_PREFIX = "data/A/"

# 模型参数
MODELS = {
    "ResNet18": {
        "lr": 0.005,
        "epochs": 5,
    },
    "EfficientNet-B0": {
        "lr": 0.003,
        "epochs": 5,
    }
}

# 数据处理参数
BATCH_SIZE = 20
NUM_WORKERS = {
    "train": 20,
    "val": 10,
    "test": 10
}
PIN_MEMORY = True
IMG_SIZE = (256, 256)

# 标签映射
LABEL_MAPPING = {
    '高风险': 0,
    '中风险': 1,
    '低风险': 2,
    '无风险': 3,
    '非楼道': 4
}

# 交叉验证
CV_SPLITS = 5
CV_RANDOM_STATE = 233

# 模型保存
MODEL_SAVE_DIR = "models"

# W&B配置
USE_WANDB = False
WANDB_PROJECT = "galaxy-classification"
WANDB_RUN_NAME = "experiment-1"

# 图像归一化参数
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225] 