"""
数据加载器模块
封装数据加载和预处理相关功能
"""

import pandas as pd
import torch
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

def get_train_transforms():
    """获取训练数据的transforms
    
    Returns:
        transforms.Compose: 训练数据预处理流程
    """
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

def get_val_transforms():
    """获取验证/测试数据的transforms
    
    Returns:
        transforms.Compose: 验证/测试数据预处理流程
    """
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

def get_train_loader(train_df, train_idx):
    """创建训练数据加载器
    
    Args:
        train_df (pd.DataFrame): 训练数据
        train_idx (np.array): 训练集索引
        
    Returns:
        DataLoader: 训练数据加载器
    """
    train_loader = torch.utils.data.DataLoader(
        GalaxyDataset(
            train_df[0].iloc[train_idx].values, 
            train_df[1].iloc[train_idx].values,
            get_train_transforms()
        ), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
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