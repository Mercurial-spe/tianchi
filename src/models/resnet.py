"""
ResNet模型实现
"""

import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    """ResNet18模型"""
    
    def __init__(self, num_classes=5, pretrained=True):
        """初始化ResNet18模型
        
        Args:
            num_classes (int): 类别数量
            pretrained (bool): 是否使用预训练权重
        """
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.model(x) 