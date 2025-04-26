# Weights & Biases 集成方案

## 一、W&B 集成目标

1. 可视化训练过程与重要指标
2. 监控训练参数（学习率、损失函数、准确度等）
3. 实现基于验证集性能的最佳模型保存

## 二、安装与初始化

```python
# 安装W&B
!pip install wandb

# 在代码开头引入并初始化
import wandb

# 初始化项目，设置项目名称和配置参数
wandb.init(
    project="tianchi-fire-safety",
    name="experiment-1",  # 可以设置不同的实验名称
    config={
        "model": "resnet18",  # 或其他模型名称
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.005,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    }
)
```

## 三、记录训练过程中的关键指标

### 修改训练循环

```python
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换到训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        
        # 计算输出
        output = model(input)
        loss = criterion(output, target)
        
        # 计算梯度并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计准确率
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 累计损失
        running_loss += loss.item()
        
        # 每10个批次记录一次进度
        if (i + 1) % 10 == 0:
            # 计算批次准确率和损失
            batch_acc = 100.0 * correct / total
            batch_loss = running_loss / (i + 1)
            
            # 记录到W&B
            wandb.log({
                "epoch": epoch,
                "batch": i,
                "train_loss": batch_loss,
                "train_acc": batch_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            print(f'Epoch: {epoch} [{i+1}/{len(train_loader)}] | Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%')
```

### 修改验证函数

```python
def validate(val_loader, model, criterion, epoch):
    # 切换到评估模式
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            
            # 计算输出
            output = model(input)
            loss = criterion(output, target)
            
            # 统计准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 累计损失
            val_loss += loss.item()
    
    # 计算验证集的准确率和损失
    val_acc = 100.0 * correct / total
    val_loss = val_loss / len(val_loader)
    
    # 记录到W&B
    wandb.log({
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc
    })
    
    print(f'Validation Epoch: {epoch} | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
    
    return val_acc
```

## 四、自动保存最佳模型

```python
def train_model_with_wandb(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    best_model_path = "best_model.pth"
    
    # 创建模型检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}/{num_epochs-1}')
        
        # 训练一个epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        # 在验证集上评估
        val_acc = validate(val_loader, model, criterion, epoch)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存模型状态
            torch.save(model.state_dict(), os.path.join("checkpoints", best_model_path))
            # 记录最佳模型到W&B
            wandb.log({"best_val_acc": best_acc})
            # 上传最佳模型到W&B
            wandb.save(os.path.join("checkpoints", best_model_path))
            print(f'Best validation accuracy: {best_acc:.2f}% - Model saved!')
    
    # 结束W&B会话
    wandb.finish()
    
    return model
```

## 五、修改原有训练代码

```python
# 主要训练脚本
model = get_model_improved().cuda()  # 使用改进的模型
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), wandb.config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 启用混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 使用修改后的训练函数
train_model_with_wandb(model, train_loader, val_loader, criterion, optimizer, num_epochs=wandb.config.epochs)
```

## 六、高级W&B功能

### 1. 记录模型架构图

```python
from torchviz import make_dot

# 记录模型架构
example_input = torch.rand(1, 3, 256, 256).cuda()
wandb.watch(model, log="all", log_freq=100)

# 创建计算图并保存
dot = make_dot(model(example_input), params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('model_architecture')
wandb.log({"model_graph": wandb.Image('model_architecture.png')})
```

### 2. 记录样本预测结果

```python
def log_predictions(model, val_loader, num_samples=10):
    model.eval()
    images, labels, preds = [], [], []
    
    with torch.no_grad():
        for img, label in val_loader:
            if len(images) >= num_samples:
                break
                
            img = img.cuda()
            output = model(img)
            _, predicted = output.max(1)
            
            for i in range(min(len(img), num_samples - len(images))):
                images.append(img[i].cpu())
                labels.append(label[i].item())
                preds.append(predicted[i].item())
    
    # 将数值标签转换为类别名称
    class_names = ["高风险", "中风险", "低风险", "无风险", "非楼道"]
    true_labels = [class_names[l] for l in labels]
    predicted_labels = [class_names[p] for p in preds]
    
    # 记录预测结果
    wandb.log({
        "predictions": [
            wandb.Image(
                img.permute(1, 2, 0).numpy(),
                caption=f"True: {true}, Pred: {pred}"
            )
            for img, true, pred in zip(images, true_labels, predicted_labels)
        ]
    })
```

### 3. 实验超参数探索

```python
# 可以使用W&B Sweep进行超参数搜索
sweep_config = {
    'method': 'bayes',  # 贝叶斯优化
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [16, 32, 64]},
        'model': {'values': ['resnet18', 'efficientnet_b0', 'efficientnet_b4']}
    }
}

sweep_id = wandb.sweep(sweep_config, project="tianchi-fire-safety")

# 定义sweep运行函数
def sweep_train():
    # 初始化wandb
    wandb.init()
    
    # 根据sweep参数选择模型
    if wandb.config.model == 'resnet18':
        model = get_model1().cuda()
    elif wandb.config.model == 'efficientnet_b0':
        model = get_model3().cuda()
    else:
        model = get_model_improved().cuda()
    
    # 训练模型...
    # (与上面的训练代码类似)

# 启动sweep
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 七、示例：完整训练脚本

```python
import os
import wandb
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast, GradScaler

# 初始化W&B
wandb.init(
    project="tianchi-fire-safety",
    name="resnet50-cosine-lr",
    config={
        "model": "resnet50",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.003,
        "optimizer": "Adam",
        "loss_function": "FocalLoss",
        "scheduler": "CosineAnnealingLR"
    }
)

# 改进的模型定义
def get_model_improved():
    model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model

# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 加载数据...
# train_loader, val_loader = ...

# 创建模型和优化器
model = get_model_improved().cuda()
wandb.watch(model, log="all")  # 监控模型参数和梯度

criterion = FocalLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = GradScaler()

# 训练循环
best_acc = 0.0
for epoch in range(wandb.config.epochs):
    # 训练模式
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 混合精度训练
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 更新权重
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        
        # 记录批次级别指标
        if (i + 1) % 10 == 0:
            wandb.log({
                "batch": epoch * len(train_loader) + i,
                "train_batch_loss": loss.item(),
                "train_batch_acc": 100 * predicted.eq(targets).sum().item() / targets.size(0),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    # 更新学习率
    scheduler.step()
    
    # 计算训练集统计
    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # 验证模式
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    
    # 计算验证集统计
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # 记录epoch级别指标
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })
    
    print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        wandb.save('best_model.pth')
        wandb.log({"best_val_acc": best_acc})
        print(f'Model saved! Best accuracy: {best_acc:.2f}%')

# 完成训练，关闭W&B
wandb.finish()
```
```