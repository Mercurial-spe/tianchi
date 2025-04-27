"""
训练器模块
封装模型训练和验证的逻辑
"""

import os
import torch
import torch.nn as nn
import wandb
import copy
from src.utils import validate
from src.config import *
from src.data_loader import mixup_data, mixup_criterion
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

class ModelTrainer:
    """模型训练器类
    封装模型训练、验证和保存的逻辑
    """
    
    def __init__(self, model_name, model, train_loader, val_loader, 
                 use_mixup=False, mixup_alpha=1.0,
                 scheduler_type=None):
        """初始化训练器
        
        Args:
            model_name (str): 模型名称
            model (nn.Module): 要训练的模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            use_mixup (bool): 是否使用Mixup
            mixup_alpha (float): Mixup的alpha参数
            scheduler_type (str): 学习率调度器类型，可选['onecycle', 'cosine', None]
        """
        self.model_name = model_name
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.best_acc = 0.0
        
        # Mixup配置
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # 创建模型保存目录
        self.save_path = os.path.join(MODEL_SAVE_DIR, model_name)
        os.makedirs(self.save_path, exist_ok=True)
        
        # 获取模型配置
        self.config = MODELS.get(model_name, {"lr": 0.001, "epochs": 5, "weight_decay": 0.0})
        self.learning_rate = self.config["lr"]
        self.epochs = self.config["epochs"]
        
        # 添加权重衰减配置
        self.weight_decay = self.config.get("weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 早停机制配置
        self.use_early_stopping = USE_EARLY_STOPPING
        self.patience = PATIENCE
        self.min_delta = MIN_DELTA
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # 学习率调度器
        self.scheduler_type = scheduler_type
        self.scheduler = None
        
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,  # 最大学习率为基础学习率的10倍
                steps_per_epoch=len(self.train_loader),
                epochs=self.epochs,
                pct_start=0.3  # 前30%的步骤提升学习率，后70%下降
            )
        elif scheduler_type == 'cosine':
            # T_0是重启周期，T_mult是周期长度乘子
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,  # 每5个epoch重启一次
                T_mult=1,  # 重启后周期保持不变
                eta_min=self.learning_rate / 100  # 最小学习率
            )
        
    def train(self):
        """训练模型并记录结果
        
        Returns:
            float: 最佳验证准确率
        """
        print(f"\n开始训练 {self.model_name} 模型")
        print(f"学习率: {self.learning_rate}, 权重衰减: {self.weight_decay}, 训练轮数: {self.epochs}")
        
        if self.use_mixup:
            print(f"使用Mixup数据增强, alpha={self.mixup_alpha}")
        
        if self.scheduler_type:
            print(f"使用学习率调度: {self.scheduler_type}")
            
        if self.use_early_stopping:
            print(f"使用早停, 耐心值: {self.patience}, 最小改进: {self.min_delta}")
        
        # 保存初始权重用于后续重新加载
        best_model_state = copy.deepcopy(self.model.state_dict())
        
        # 配置W&B
        if USE_WANDB:
            config = {
                "model": self.model_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "batch_size": self.train_loader.batch_size,
                "optimizer": "Adam",
                "use_mixup": self.use_mixup,
                "mixup_alpha": self.mixup_alpha if self.use_mixup else None,
                "scheduler": self.scheduler_type,
                "early_stopping": self.use_early_stopping,
                "patience": self.patience if self.use_early_stopping else None
            }
            wandb.config.update(config)
        
        # 训练循环
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch+1}/{self.epochs}')
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            
            # 验证阶段
            val_acc, val_loss, self.best_acc, is_best = validate(
                self.val_loader, self.model, self.criterion, 
                epoch=epoch, optimizer=self.optimizer, 
                best_acc=self.best_acc, save_path=self.save_path
            )
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 最佳准确率: {self.best_acc:.4f}")
            
            # 如果是最佳模型，保存状态
            if is_best:
                best_model_state = copy.deepcopy(self.model.state_dict())
            
            # 早停检查
            if self.use_early_stopping:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
                    
                    if self.early_stopping_counter >= self.patience:
                        print(f"早停触发！验证损失 {self.patience} 轮未改善")
                        break
            
            # 余弦退火学习率调整在epoch结束时更新
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
                
            # 记录到W&B
            if USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_acc": self.best_acc,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "early_stop_counter": self.early_stopping_counter if self.use_early_stopping else 0
                })
        
        # 训练结束，加载最佳模型状态
        self.model.load_state_dict(best_model_state)
        
        print(f"{self.model_name} 训练完成。最佳验证准确率: {self.best_acc:.4f}")
        return self.best_acc
    
    def _train_epoch(self, epoch):
        """训练一个epoch
        
        Args:
            epoch (int): 当前epoch
            
        Returns:
            tuple: (avg_loss, avg_acc) 平均损失和准确率
        """
        # 切换到训练模式
        self.model.train()

        end = 0
        total_loss = 0.0
        total_acc = 0
        for i, data in enumerate(self.train_loader):
            # 确保兼容数据加载器返回多个值的情况
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                input = data[0].cuda(non_blocking=True)
                target = data[1].cuda(non_blocking=True)
            else:
                raise ValueError("数据加载器格式不正确，无法获取输入和目标")

            # 如果使用Mixup
            if self.use_mixup:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup_alpha)
                output = self.model(input)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                
                # Mixup下准确率计算近似值
                acc1 = lam * (output.argmax(1).long() == target_a.long()).sum().item() + \
                       (1 - lam) * (output.argmax(1).long() == target_b.long()).sum().item()
            else:
                # 常规训练
                output = self.model(input)
                loss = self.criterion(output, target)
                acc1 = (output.argmax(1).long() == target.long()).sum().item()
                
            total_acc += acc1
            total_loss += loss.item()

            # 计算梯度并更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # OneCycle学习率需要每batch更新
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()

            if i % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item():.4f}, Acc: {acc1 / input.size(0):.4f}, LR: {current_lr:.6f}")
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader.dataset)
        
        return avg_loss, avg_acc

    def progressive_train(self, img_sizes=[160, 192, 224]):
        """使用渐进式图像大小进行训练
        
        Args:
            img_sizes (list): 逐渐增加的图像大小列表
        
        Returns:
            float: 最佳验证准确率
        """
        raise NotImplementedError("渐进式训练尚未实现") 