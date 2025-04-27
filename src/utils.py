import numpy as np
import cv2
import os
import time
from datetime import datetime
import torch
import torch.nn as nn

def is_valid_image(image_path):
    if not os.path.exists(image_path):
        return False
    image = cv2.imread(image_path)
    return image is not None

def save_model(model, epoch, val_acc, optimizer, save_path="models"):
    """保存模型检查点
    
    Args:
        model: PyTorch模型
        epoch: 当前epoch
        val_acc: 验证准确率
        optimizer: 优化器
        save_path: 保存路径
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 使用固定文件名保存最佳模型
    filename = f"{save_path}/best_model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, filename)
    print(f"Best model saved to {filename} with accuracy: {val_acc:.4f} at epoch {epoch}")
    return filename

def validate(val_loader, model, criterion, epoch=-1, optimizer=None, best_acc=0.0, save_path="models"):
    """验证模型性能
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        criterion: 损失函数
        epoch: 当前epoch
        optimizer: 优化器
        best_acc: 当前最佳准确率
        save_path: 模型保存路径
        
    Returns:
        current_acc: 当前准确率
        avg_loss: 平均损失
        best_acc: 更新后的最佳准确率
        is_best: 是否是最佳模型
    """
    # 切换到评估模式
    model.eval()

    total_acc = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # 计算输出
            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item()

            # 计算准确率
            total_acc += (output.argmax(1).long() == target.long()).sum().item()
    
    current_acc = total_acc / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)
    
    # 如果当前准确率更好，保存模型
    is_best = False
    if current_acc > best_acc and optimizer is not None and epoch >= 0:
        is_best = True
        best_acc = current_acc
        save_model(model, epoch, current_acc, optimizer, save_path)
    
    return current_acc, avg_loss, best_acc, is_best 