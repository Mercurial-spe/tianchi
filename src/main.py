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

from utils import is_valid_image, train, validate, predict
from models import get_model1, get_model3
from dataset import GalaxyDataset
def main():
    # 数据准备
    train_df = pd.read_csv("data/train.txt", sep="\t", header=None)
    train_df[0] = 'data/train/' + train_df[0]
    print(train_df)
    train_df = train_df[train_df[0].apply(is_valid_image)]
    
    # 显示不同类别的样本图片
    for lbl in train_df[1].value_counts().index:
        img = cv2.imread(train_df[train_df[1] == lbl][0].sample(1).values[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
    
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
    
    # 训练第一个模型
    model = get_model1().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.005)
    best_acc = 0.0
    for epoch in range(5):
        print('Epoch: ', epoch)
    
        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)
        print("Val acc", val_acc)
    
    # 训练第二个模型
    model = get_model3().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.003)
    best_acc = 0.0
    for epoch in range(5):
        print('Epoch: ', epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)
        print("Val acc", val_acc)
    
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
    
    # 预测并生成提交文件
    pred = predict(test_loader, model, criterion)
    pred = np.stack(pred)
    inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}
    inverse_transform = np.vectorize(inverse_mapping_dict.get)
    test_df["label"] = inverse_transform(pred)
    test_df[[0, "label"]].to_csv("submit.txt", index=None, sep="\t", header=None) 

if __name__ == "__main__":
    main()