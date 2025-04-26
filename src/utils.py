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

def load_model(model, optimizer, checkpoint_path):
    """加载模型检查点
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
    
    Returns:
        epoch: 加载的epoch
        best_acc: 最佳验证准确率
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    print(f"Model loaded from {checkpoint_path} (epoch: {epoch}, acc: {val_acc:.4f})")
    return epoch, val_acc

def validate(val_loader, model, criterion, epoch=-1, optimizer=None, best_acc=0.0, save_path="models"):
    # switch to evaluate mode
    model.eval()

    total_acc = 0
    total_loss = 0.0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item()

            # measure accuracy and record loss
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

def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    end = time.time()
    total_loss = 0.0
    total_acc = 0
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        acc1 = (output.argmax(1).long() == target.long()).sum().item()
        total_acc += acc1
        total_loss += loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(datetime.now(), loss.item(), acc1 / input.size(0))
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader.dataset)
    
    return avg_loss, avg_acc

def predict(test_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    pred = [] 
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            pred += list(output.argmax(1).long().cpu().numpy())
    return pred 