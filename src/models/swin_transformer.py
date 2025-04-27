"""
Swin Transformer模型实现
"""

import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    """Swin Transformer模型，使用timm库实现"""
    
    def __init__(self, num_classes=5, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        """初始化Swin Transformer模型
        
        Args:
            num_classes (int): 类别数量
            model_name (str): 模型名称，支持swin_tiny_patch4_window7_224等
            pretrained (bool): 是否使用预训练权重
        """
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.model(x) 