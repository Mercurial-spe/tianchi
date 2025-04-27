"""
预测器模块
封装模型预测和结果生成的逻辑
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from src.utils import predict
from src.config import *

class ModelPredictor:
    """模型预测器类
    封装模型预测和结果生成的逻辑
    """
    
    def __init__(self, model, test_loader):
        """初始化预测器
        
        Args:
            model (nn.Module): 预测模型
            test_loader (DataLoader): 测试数据加载器
        """
        self.model = model.cuda()
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().cuda()
        
    def load_best_model(self, model_name):
        """加载最佳模型
        
        Args:
            model_name (str): 模型名称
            
        Returns:
            bool: 是否成功加载模型
        """
        best_model_path = os.path.join(MODEL_SAVE_DIR, model_name, "best_model.pth")
        
        if os.path.exists(best_model_path):
            # 加载最佳模型
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型: {best_model_path}")
            print(f"模型信息: 准确率 {checkpoint['val_acc']:.4f}, Epoch {checkpoint['epoch']}")
            return True
        else:
            print(f"警告: 未找到最佳模型 {best_model_path}，使用默认模型。")
            return False
    
    def predict(self):
        """执行预测
        
        Returns:
            np.array: 预测结果
        """
        print("开始预测...")
        pred = predict(self.test_loader, self.model, self.criterion)
        pred = np.stack(pred)
        print(f"预测完成，共 {len(pred)} 个样本")
        return pred
    
    def generate_submission(self, pred, output_file="submit.txt"):
        """生成提交文件
        
        Args:
            pred (np.array): 预测结果
            output_file (str, optional): 输出文件名. 默认为 "submit.txt".
            
        Returns:
            str: 提交文件路径
        """
        # 加载测试数据
        test_df = pd.read_csv(TEST_DATA_PATH, sep="\t", header=None)
        
        # 将数字标签转换为文本标签
        inverse_mapping_dict = {v: k for k, v in LABEL_MAPPING.items()}
        inverse_transform = np.vectorize(inverse_mapping_dict.get)
        test_df["label"] = inverse_transform(pred)
        
        # 保存提交文件
        test_df[[0, "label"]].to_csv(output_file, index=None, sep="\t", header=None)
        print(f"提交文件已保存为 {output_file}")
        
        # 显示类别分布
        print("预测结果类别分布:")
        for label, count in test_df["label"].value_counts().items():
            print(f"  {label}: {count}个样本 ({count/len(test_df)*100:.1f}%)")
            
        return output_file
    
    def predict_and_submit(self, output_file="submit.txt"):
        """预测并生成提交文件
        
        Args:
            output_file (str, optional): 输出文件名. 默认为 "submit.txt".
            
        Returns:
            str: 提交文件路径
        """
        pred = self.predict()
        return self.generate_submission(pred, output_file) 