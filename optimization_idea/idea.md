# 消防隐患识别比赛上分攻略

## 一、模型优化方向

### 1. 基础模型升级
- 从ResNet18升级到更强大的模型
```python
# 替换为ResNet50/101或EfficientNet-B3/B4
def get_model_improved():
    model = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    return model
```

### 2. 损失函数优化
- 使用Focal Loss解决类别不平衡问题
```python
# 实现Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 使用方法
criterion = FocalLoss().cuda()
```

### 3. 学习率策略优化
- 实现学习率调整策略
```python
# 使用学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 在训练循环中添加
for epoch in range(10):
    train(train_loader, model, criterion, optimizer, epoch)
    scheduler.step()
    val_acc = validate(val_loader, model, criterion)
```

## 二、数据增强策略

### 1. 高级数据增强
- 使用Albumentations库实现更强的数据增强
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomResizedCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.ColorJitter(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

### 2. 测试时增强(TTA)
- 实现测试时数据增强提高预测准确率
```python
def tta_predict(test_loader, model, n_augmentations=5):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            batch_preds = []
            inputs = inputs.cuda()
            
            # 原始预测
            outputs = model(inputs)
            batch_preds.append(outputs.softmax(dim=1))
            
            # 水平翻转
            outputs = model(torch.flip(inputs, dims=[3]))
            batch_preds.append(outputs.softmax(dim=1))
            
            # 更多增强...
            
            # 平均所有预测
            batch_preds = torch.stack(batch_preds).mean(dim=0)
            predictions.append(batch_preds.cpu())
    
    return torch.cat(predictions)
```

## 三、模型集成方法

### 1. 多模型集成
- 使用不同架构模型进行集成
```python
models = [
    get_model1().cuda(),  # ResNet18
    get_model3().cuda(),  # EfficientNet-B0
    get_model_improved().cuda()  # 更强的模型
]

# 训练所有模型
for model in models:
    train_model(model, train_loader, val_loader, criterion, optimizer)

# 预测时集成
def ensemble_predict(test_loader, models):
    predictions = []
    for model in models:
        model.eval()
        model_preds = predict(test_loader, model)
        predictions.append(model_preds)
    
    # 投票或平均
    final_preds = np.stack(predictions).mean(axis=0).argmax(axis=1)
    return final_preds
```

### 2. K折交叉验证集成
- 利用多折模型集成提高稳定性
```python
k_folds = 5
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_models = []
for fold, (train_idx, val_idx) in enumerate(kf.split(df["path"], df["label"])):
    model = get_model_improved().cuda()
    # 训练模型
    # ...
    fold_models.append(model)

# 集成预测
ensemble_pred = ensemble_predict(test_loader, fold_models)
```

## 四、优化训练策略

### 1. 混合精度训练
- 加速训练并节省显存
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中
for i, (input, target) in enumerate(train_loader):
    optimizer.zero_grad()
    
    with autocast():
        output = model(input)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 渐进式学习率
- 先冻结预训练层，后微调全网络
```python
# 第一阶段：只训练分类头
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# 使用较大学习率训练几个轮次
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
for epoch in range(3):
    train(train_loader, model, criterion, optimizer, epoch)

# 第二阶段：微调整个网络
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5):
    train(train_loader, model, criterion, optimizer, epoch)
```

## 五、工程实现技巧

### 1. 图像预处理优化
- 使用更合适的输入尺寸
```python
# 更大的图像尺寸，保留更多细节
transforms.Resize((320, 320)) 或 transforms.Resize((384, 384))
```

### 2. 伪标签技术
- 利用测试集提高模型表现
```python
# 第一阶段：用初始模型预测测试集
test_pred = predict(test_loader, model)
confident_idx = np.where(test_pred.max(axis=1) > 0.9)[0]  # 高置信度样本

# 第二阶段：将高置信度样本加入训练
pseudo_dataset = torch.utils.data.ConcatDataset([
    train_dataset,
    GalaxyDataset(test_df["path"].values[confident_idx], 
                 test_pred.argmax(axis=1)[confident_idx], transform)
])

# 使用扩充的数据集重新训练
pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=32, shuffle=True)
train(pseudo_loader, model, criterion, optimizer, epoch)
```
```