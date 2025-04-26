import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
import wandb

from src.utils import is_valid_image, train, validate, predict, save_model, load_model
from src.models import get_model1, get_model3
from src.dataset import GalaxyDataset

def main():
    # 初始化W&B（注意：首次使用需要先运行pip install wandb并登录）
    use_wandb = False  # 设置为True启用W&B
    if use_wandb:
        wandb.init(project="galaxy-classification", name="experiment-1")
        
    # 数据准备
    train_df = pd.read_csv("data/train.txt", sep="\t", header=None)
    train_df[0] = 'data/train/' + train_df[0]
    print(train_df)
    train_df = train_df[train_df[0].apply(is_valid_image)]
    
    # 显示不同类别的样本图片
    # for lbl in train_df[1].value_counts().index:
    #     img = cv2.imread(train_df[train_df[1] == lbl][0].sample(1).values[0])
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.figure()
    #     plt.imshow(img)
    
    # 标签映射
    mapping_dict = {
        '高风险': 0,
        '中风险': 1,
        '低风险': 2,
        '无风险': 3,
        '非楼道': 4
    }
    
    train_df[1] = train_df[1].map(mapping_dict)
    
    # 数据集划分
    skf = StratifiedKFold(n_splits=5, random_state=233, shuffle=True)
    
    for _, (train_idx, val_idx) in enumerate(skf.split(train_df[0].values, train_df[1].values)):
        break
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        GalaxyDataset(train_df[0].iloc[train_idx].values, train_df[1].iloc[train_idx].values,
                transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=20, shuffle=True, num_workers=20, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        GalaxyDataset(train_df[0].iloc[val_idx].values, train_df[1].iloc[val_idx].values,
                transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=20, shuffle=False, num_workers=10, pin_memory=True
    )
    
    # 创建保存模型的目录
    os.makedirs("models", exist_ok=True)
    
    # 训练第一个模型
    train_model(
        model_name="ResNet18",
        model=get_model1().cuda(),
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.005,
        epochs=5,
        use_wandb=use_wandb
    )
    
    # 训练第二个模型
    train_model(
        model_name="EfficientNet-B0",
        model=get_model3().cuda(),
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.003,
        epochs=5,
        use_wandb=use_wandb
    )
    
    # 测试数据处理
    test_df = pd.read_csv("data/A.txt", sep="\t", header=None)
    test_df["path"] = 'data/A/' + test_df[0]
    test_df["label"] = 1
    
    test_loader = torch.utils.data.DataLoader(
        GalaxyDataset(test_df["path"].values, test_df["label"].values,
                transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=20, shuffle=False, num_workers=10, pin_memory=True
    )
    
    # 加载最佳模型进行预测
    best_model = get_model3().cuda()  # 假设我们使用EfficientNet-B0作为最终模型
    best_model_path = os.path.join("models", "EfficientNet-B0", "best_model.pth")
    
    if os.path.exists(best_model_path):
        # 加载最佳模型
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_model_path} with accuracy {checkpoint['val_acc']:.4f} (epoch {checkpoint['epoch']})")
    else:
        print(f"Warning: Best model not found at {best_model_path}. Using default model.")
    
    # 预测并生成提交文件
    criterion = nn.CrossEntropyLoss().cuda()
    pred = predict(test_loader, best_model, criterion)
    pred = np.stack(pred)
    inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}
    inverse_transform = np.vectorize(inverse_mapping_dict.get)
    test_df["label"] = inverse_transform(pred)
    test_df[[0, "label"]].to_csv("submit.txt", index=None, sep="\t", header=None)

def train_model(model_name, model, train_loader, val_loader, learning_rate, epochs, use_wandb=False):
    """训练模型并记录结果
    
    Args:
        model_name: 模型名称，用于日志
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        learning_rate: 学习率
        epochs: 训练轮数
        use_wandb: 是否使用W&B记录
    """
    print(f"\n开始训练 {model_name} 模型")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # 配置W&B
    if use_wandb:
        config = {
            "model": model_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam"
        }
        wandb.config.update(config)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        
        # 训练阶段
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 验证阶段
        val_acc, val_loss, best_acc, is_best = validate(
            val_loader, model, criterion, 
            epoch=epoch, optimizer=optimizer, 
            best_acc=best_acc, save_path=f"models/{model_name}"
        )
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Acc: {best_acc:.4f}")
        
        # 记录到W&B
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc
            })
    
    print(f"{model_name} 训练完成。最佳验证准确率: {best_acc:.4f}")
    return best_acc

if __name__ == "__main__":
    main()