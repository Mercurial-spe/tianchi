"""
EfficientNet模型实现
"""

import torch.nn as nn
import torchvision.models as models

class EfficientNet(nn.Module):
    """EfficientNet模型"""
    
    def __init__(self, num_classes=5, model_name='efficientnet_b0', pretrained=True):
        """初始化EfficientNet模型
        
        Args:
            num_classes (int): 类别数量
            model_name (str): 模型名称，如'efficientnet_b0'
            pretrained (bool): 是否使用预训练权重
        """
        super(EfficientNet, self).__init__()
        
        # 获取模型实例
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif model_name == 'efficientnet_b2':
            self.model = models.efficientnet_b2(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(1408, num_classes)
        elif model_name == 'efficientnet_b3':
            self.model = models.efficientnet_b3(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(1536, num_classes)
        else:
            raise ValueError(f"不支持的EfficientNet模型: {model_name}")
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.model(x)