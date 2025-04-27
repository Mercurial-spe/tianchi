"""
Swin Transformer模型实现
"""

import torch.nn as nn
import timm
from safetensors.torch import load_file
import torch
import os

class SwinTransformer(nn.Module):
    """Swin Transformer模型，使用timm库实现"""
    
    def __init__(self, num_classes=5, model_name='swin_tiny_patch4_window7_224', pretrained=False, pretrained_path=None):
        """初始化Swin Transformer模型
        
        Args:
            num_classes (int): 类别数量
            model_name (str): 模型名称，支持swin_tiny_patch4_window7_224等
            pretrained (bool): 是否使用预训练权重(从网络下载)
            pretrained_path (str): 本地预训练权重路径，如果提供则优先使用
        """
        super(SwinTransformer, self).__init__()
        # 首先创建没有预训练权重的模型
        self.model = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=num_classes
        )
        
        # 如果提供了本地预训练权重路径，则加载本地权重
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_weights(pretrained_path)
        # 否则，如果pretrained=True，则尝试从网络加载预训练权重
        elif pretrained:
            # 重新创建模型并加载预训练权重
            self.model = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=num_classes
            )
    
    def _load_pretrained_weights(self, pretrained_path):
        """从本地加载预训练权重
        
        Args:
            pretrained_path (str): 预训练权重文件路径
        """
        print(f"从本地路径加载模型权重: {pretrained_path}")
        try:
            if pretrained_path.endswith('.safetensors'):
                # 使用safetensors加载权重
                state_dict = load_file(pretrained_path)
            else:
                # 对于PyTorch权重，使用torch.load并处理可能的错误
                try:
                    state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                except Exception as e:
                    print(f"使用weights_only=False加载失败: {str(e)}")
                    state_dict = torch.load(pretrained_path, map_location='cpu')
            
            # 过滤掉分类器层的权重
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # 排除分类器层的权重
                if 'classifier' in k or 'head' in k or 'fc' in k:
                    print(f"跳过加载分类器层权重: {k} 形状: {v.shape}")
                    continue
                filtered_state_dict[k] = v
            
            # 加载权重到模型
            result = self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"权重加载结果 - 缺少的键: {result.missing_keys}")
            print(f"权重加载结果 - 未使用的键: {result.unexpected_keys}")
            
            # 验证权重是否加载成功
            non_zero_weights = 0
            total_weights = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    total_weights += 1
                    if torch.sum(torch.abs(param) > 1e-6) > 0:
                        non_zero_weights += 1
            
            percentage = (non_zero_weights / total_weights) * 100 if total_weights > 0 else 0
            print(f"非零权重比例: {percentage:.2f}% ({non_zero_weights}/{total_weights})")
            
        except Exception as e:
            print(f"加载预训练权重失败: {str(e)}")
            print("将使用随机初始化的权重")
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.model(x) 