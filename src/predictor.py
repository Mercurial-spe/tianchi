"""
预测器模块
封装模型预测和结果生成的功能
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

class ModelPredictor:
    """模型预测器类
    封装模型加载、预测和结果生成的功能
    """
    
    def __init__(self, model, test_loader):
        """初始化预测器
        
        Args:
            model (nn.Module): 模型实例
            test_loader (DataLoader): 测试数据加载器
        """
        self.model = model.cuda()
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        
    def load_best_model(self, model_name):
        """加载指定模型的最佳权重
        
        Args:
            model_name (str): 模型名称
        """
        model_path = f"models/{model_name}/best_model.pth"
        if not os.path.exists(model_path):
            print(f"警告: 未找到最佳模型权重 {model_path}，使用初始模型")
            return False
        
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型权重: {model_path} (准确率: {checkpoint['val_acc']:.4f}, Epoch: {checkpoint['epoch']})")
        return True
        
    def predict(self):
        """对测试集进行预测
        
        Returns:
            list: 预测的类别索引列表
        """
        # 切换到评估模式
        self.model.eval()
        
        predictions = []
        img_paths = []
        
        # 禁用梯度计算
        with torch.no_grad():
            # 使用tqdm显示进度条
            for i, (inputs, targets, paths) in enumerate(tqdm(self.test_loader, desc="测试预测")):
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                # 获取模型输出
                outputs = self.model(inputs)
                
                # 获取预测的类别索引
                _, preds = torch.max(outputs, 1)
                
                # 收集结果
                predictions.extend(preds.cpu().numpy())
                img_paths.extend(paths)
        
        return predictions, img_paths
    
    def predict_and_submit(self, output_path=None):
        """预测并生成提交文件
        
        Args:
            output_path (str, optional): 输出文件路径，默认为带有时间戳的文件名
        """
        # 生成默认输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"submission_{timestamp}.txt"
            
        # 开始计时
        start_time = time.time()
        
        # 进行预测
        predictions, img_paths = self.predict()
        
        # 类别索引到标签名称的映射
        idx_to_label = {v: k for k, v in {
            '高风险': 0,
            '中风险': 1,
            '低风险': 2,
            '无风险': 3,
            '非楼道': 4
        }.items()}
        
        # 创建提交文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (pred, path) in enumerate(zip(predictions, img_paths)):
                # 从路径中提取图像文件名
                img_name = os.path.basename(path)
                # 获取预测的标签名称
                label = idx_to_label[pred]
                # 写入提交格式
                f.write(f"{img_name}\t{label}\n")
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        print(f"预测完成，生成提交文件: {output_path} (耗时: {elapsed_time:.2f}秒)")
        return output_path 