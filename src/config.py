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
        "lr": 0.003,
        "epochs": 15,
        "weight_decay": 1e-4,
        "batch_size": 32
    },
    "EfficientNet-B0": {
        "lr": 0.002,
        "epochs": 15,
        "weight_decay": 1e-5,
        "batch_size": 24
    },
    "EfficientNetV2-S": {
        "lr": 0.001,
        "epochs": 15,
        "weight_decay": 5e-6,
        "batch_size": 16
    },
    "Swin-T": {
        "lr": 0.003,
        "epochs": 15,
        "img_size": (224, 224),  # Swin Transformer需要224x224的输入尺寸
        "weight_decay": 1e-6,
        "batch_size": 16
    }
}

# 数据处理参数
BATCH_SIZE = 32  # 默认批次大小，会被模型特定配置覆盖
NUM_WORKERS = {
    "train": 20,
    "val": 10,
    "test": 10
}
PIN_MEMORY = True
IMG_SIZE = (256, 256)  # 默认图像尺寸

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
USE_WANDB = True
WANDB_PROJECT = "galaxy-classification"
WANDB_RUN_NAME = "experiment-2"

# 图像归一化参数
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 高级功能配置
# 数据增强
USE_RAND_AUGMENT = True  # 是否使用RandAugment
USE_MIXUP = True  # 是否使用Mixup
MIXUP_ALPHA = 1.0  # Mixup的alpha参数

# 学习率调度
SCHEDULER_TYPE = "cosine"  # 学习率调度器类型，可选值: "onecycle", "cosine", None

# 类别平衡
USE_BALANCED_SAMPLER = True  # 是否使用类别平衡采样器

# 渐进式训练
USE_PROGRESSIVE_RESIZING = False  # 是否使用渐进式图像大小训练
PROGRESSIVE_SIZES = [160, 192, 224]  # 渐进式训练图像大小列表

# 早停策略
USE_EARLY_STOPPING = True  # 启用早停机制
PATIENCE = 3  # 早停耐心值，连续几轮没有改进则停止训练
MIN_DELTA = 0.001  # 最小改进阈值 