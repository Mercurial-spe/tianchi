"""
训练器模块
封装模型训练和验证的逻辑
"""

import os
import torch
import torch.nn as nn
import wandb
from src.utils import train, validate
from src.config import *

class ModelTrainer:
    """模型训练器类
    封装模型训练、验证和保存的逻辑
    """
    
    def __init__(self, model_name, model, train_loader, val_loader):
        """初始化训练器
        
        Args:
            model_name (str): 模型名称
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
        """
        self.model_name = model_name
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.best_acc = 0.0
        
        # 创建模型保存目录
        self.save_path = os.path.join(MODEL_SAVE_DIR, model_name)
        os.makedirs(self.save_path, exist_ok=True)
        
        # 获取模型配置
        self.config = MODELS.get(model_name, {"lr": 0.001, "epochs": 5})
        self.learning_rate = self.config["lr"]
        self.epochs = self.config["epochs"]
        self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        
    def train(self):
        """训练模型并记录结果
        
        Returns:
            float: 最佳验证准确率
        """
        print(f"\n开始训练 {self.model_name} 模型")
        print(f"学习率: {self.learning_rate}, 训练轮数: {self.epochs}")
        
        # 配置W&B
        if USE_WANDB:
            config = {
                "model": self.model_name,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.train_loader.batch_size,
                "optimizer": "Adam"
            }
            wandb.config.update(config)
        
        # 训练循环
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch+1}/{self.epochs}')
            
            # 训练阶段
            train_loss, train_acc = train(
                self.train_loader, self.model, 
                self.criterion, self.optimizer, epoch
            )
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            
            # 验证阶段
            val_acc, val_loss, self.best_acc, is_best = validate(
                self.val_loader, self.model, self.criterion, 
                epoch=epoch, optimizer=self.optimizer, 
                best_acc=self.best_acc, save_path=self.save_path
            )
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 最佳准确率: {self.best_acc:.4f}")
            
            # 记录到W&B
            if USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_acc": self.best_acc
                })
        
        print(f"{self.model_name} 训练完成。最佳验证准确率: {self.best_acc:.4f}")
        return self.best_acc 