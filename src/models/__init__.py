"""
模型包
提供各种模型架构的实现
"""

from .resnet import ResNet18
from .efficientnet import EfficientNet
from .efficientnetv2 import EfficientNetV2
from .swin_transformer import SwinTransformer

def get_model(model_name, num_classes=5):
    """获取指定名称的模型实例
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 类别数量
        
    Returns:
        nn.Module: 模型实例
    """
    if model_name == "ResNet18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "EfficientNet-B0":
        return EfficientNet(num_classes=num_classes, model_name='efficientnet_b0')
    elif model_name == "EfficientNetV2-S":
        return EfficientNetV2(num_classes=num_classes, model_name='efficientnetv2_s')
    elif model_name == "Swin-T":
        return SwinTransformer(num_classes=num_classes, model_name='swin_tiny_patch4_window7_224')
    else:
        raise ValueError(f"未知模型: {model_name}") 