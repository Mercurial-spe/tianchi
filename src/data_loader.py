"""
数据加载器模块
封装数据加载和预处理相关功能
"""

import pandas as pd
import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold

from src.dataset import GalaxyDataset
from src.utils import is_valid_image
from src.config import *

def load_train_data():
    """加载训练数据并处理
    
    Returns:
        pd.DataFrame: 处理后的训练数据
    """
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep="\t", header=None)
    train_df[0] = TRAIN_IMG_PREFIX + train_df[0]
    # 过滤无效图像
    train_df = train_df[train_df[0].apply(is_valid_image)]
    # 将文本标签映射为数字
    train_df[1] = train_df[1].map(LABEL_MAPPING)
    print(f"加载训练数据: {len(train_df)}个样本")
    return train_df

def load_test_data():
    """加载测试数据并处理
    
    Returns:
        pd.DataFrame: 处理后的测试数据
    """
    test_df = pd.read_csv(TEST_DATA_PATH, sep="\t", header=None)
    test_df["path"] = TEST_IMG_PREFIX + test_df[0]
    test_df["label"] = 1  # 占位标签
    print(f"加载测试数据: {len(test_df)}个样本")
    return test_df

def get_train_val_split(train_df):
    """划分训练集和验证集
    
    Args:
        train_df (pd.DataFrame): 训练数据
        
    Returns:
        tuple: (train_idx, val_idx) 训练集和验证集的索引
    """
    skf = StratifiedKFold(n_splits=CV_SPLITS, random_state=CV_RANDOM_STATE, shuffle=True)
    
    for _, (train_idx, val_idx) in enumerate(skf.split(train_df[0].values, train_df[1].values)):
        break
    
    print(f"训练集: {len(train_idx)}个样本, 验证集: {len(val_idx)}个样本")
    return train_idx, val_idx

def get_transforms(img_size=IMG_SIZE, is_training=True, use_rand_augment=False):
    """获取数据变换，支持不同图像大小和高级增强
    
    Args:
        img_size (tuple or int): 图像大小
        is_training (bool): 是否用于训练
        use_rand_augment (bool): 是否使用RandAugment
        
    Returns:
        transforms.Compose: 数据预处理流程
    """
    if is_training:
        transform_list = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
        
        # 添加RandAugment
        if use_rand_augment:
            transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
        
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])

def get_train_transforms(use_rand_augment=False):
    """获取训练数据的transforms
    
    Args:
        use_rand_augment (bool): 是否使用RandAugment
        
    Returns:
        transforms.Compose: 训练数据预处理流程
    """
    return get_transforms(IMG_SIZE, True, use_rand_augment)

def get_val_transforms():
    """获取验证/测试数据的transforms
    
    Returns:
        transforms.Compose: 验证/测试数据预处理流程
    """
    return get_transforms(IMG_SIZE, False, False)

def mixup_data(x, y, alpha=1.0):
    """实现Mixup数据增强
    
    Args:
        x (Tensor): 输入数据
        y (Tensor): 标签
        alpha (float): beta分布参数
        
    Returns:
        tuple: 混合后的数据、标签A、标签B和lambda值
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup的损失函数
    
    Args:
        criterion: 原始损失函数
        pred: 预测输出
        y_a: 第一个标签
        y_b: 第二个标签
        lam: 混合系数
        
    Returns:
        float: 混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class BalancedSampler(torch.utils.data.sampler.Sampler):
    """类别平衡采样器，对类别不平衡数据进行采样平衡"""
    
    def __init__(self, dataset, indices=None):
        """初始化平衡采样器
        
        Args:
            dataset: 数据集
            indices: 采样索引列表，如果为None则使用全部数据
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        
        # 获取每个样本的标签
        self.labels = self.get_labels(self.dataset)
        self.label_to_indices = self.get_label_to_indices(self.labels)
        self.labels_set = list(set(self.labels))
        self.count = self.get_class_count()
        
        # 计算每个类别的权重，反比于类别数量
        self.weights = {label: 1.0/self.count[label] for label in self.labels_set}
        self.weights_sum = sum(self.weights.values())
        
        # 归一化权重
        for label in self.weights:
            self.weights[label] /= self.weights_sum
            
    def get_labels(self, dataset):
        """获取数据集中所有样本的标签"""
        return [dataset[i][1] for i in self.indices]
    
    def get_label_to_indices(self, labels):
        """构建标签到索引的映射"""
        label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(self.indices[idx])
        return label_to_indices
    
    def get_class_count(self):
        """计算每个类别的样本数量"""
        count = {}
        for label in self.labels_set:
            count[label] = len(self.label_to_indices[label])
        return count
    
    def __iter__(self):
        """迭代器，返回平衡后的样本索引"""
        samples_per_class = {label: [] for label in self.labels_set}
        num_batches = len(self) // len(self.labels_set)
        
        # 为每个类别选择样本
        for label in self.labels_set:
            indices = self.label_to_indices[label]
            samples_per_class[label].extend(np.random.choice(
                indices, 
                size=int(num_batches * self.weights[label] * len(self.labels_set)),
                replace=True
            ).tolist())
            
        # 打乱采样结果
        shuffled = []
        for i in range(num_batches):
            for label in self.labels_set:
                if i < len(samples_per_class[label]):
                    shuffled.append(samples_per_class[label][i])
                    
        return iter(shuffled)
    
    def __len__(self):
        """返回采样器的长度"""
        return len(self.indices)

def get_train_loader(train_df, train_idx, use_rand_augment=False, use_balanced_sampler=False):
    """创建训练数据加载器，支持高级增强和平衡采样
    
    Args:
        train_df (pd.DataFrame): 训练数据
        train_idx (np.array): 训练集索引
        use_rand_augment (bool): 是否使用RandAugment
        use_balanced_sampler (bool): 是否使用平衡采样器
        
    Returns:
        DataLoader: 训练数据加载器
    """
    train_dataset = GalaxyDataset(
        train_df[0].iloc[train_idx].values, 
        train_df[1].iloc[train_idx].values,
        get_train_transforms(use_rand_augment)
    )
    
    # 如果启用平衡采样器
    sampler = None
    if use_balanced_sampler:
        sampler = BalancedSampler(train_dataset)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        shuffle=sampler is None,  # 如果使用采样器，不要shuffle
        num_workers=NUM_WORKERS["train"], 
        pin_memory=PIN_MEMORY
    )
    return train_loader

def get_val_loader(train_df, val_idx):
    """创建验证数据加载器
    
    Args:
        train_df (pd.DataFrame): 训练数据
        val_idx (np.array): 验证集索引
        
    Returns:
        DataLoader: 验证数据加载器
    """
    val_loader = torch.utils.data.DataLoader(
        GalaxyDataset(
            train_df[0].iloc[val_idx].values, 
            train_df[1].iloc[val_idx].values,
            get_val_transforms()
        ), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS["val"], 
        pin_memory=PIN_MEMORY
    )
    return val_loader

def get_test_loader(test_df):
    """创建测试数据加载器
    
    Args:
        test_df (pd.DataFrame): 测试数据
        
    Returns:
        DataLoader: 测试数据加载器
    """
    test_loader = torch.utils.data.DataLoader(
        GalaxyDataset(
            test_df["path"].values, 
            test_df["label"].values,
            get_val_transforms()
        ), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS["test"], 
        pin_memory=PIN_MEMORY
    )
    return test_loader 