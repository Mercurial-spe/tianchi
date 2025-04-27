"""
模型包
提供各种模型架构的实现
"""

from .resnet import ResNet18
from .efficientnet import EfficientNet
from .efficientnetv2 import EfficientNetV2
from .swin_transformer import SwinTransformer

def get_model(model_name, num_classes=5, pretrained=True, pretrained_path=None):
    """获取指定名称的模型实例
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重(从网络下载)
        pretrained_path (str): 本地预训练权重路径，如果提供则优先使用
        
    Returns:
        nn.Module: 模型实例
    """
    if model_name == "ResNet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "EfficientNet-B0":
        return EfficientNet(num_classes=num_classes, model_name='efficientnet_b0', pretrained=pretrained)
    elif model_name == "EfficientNetV2-S":
        return EfficientNetV2(num_classes=num_classes, model_name='efficientnetv2_s', 
                            pretrained_path="pretrained_models\EfficientNetV2.safetensors")
    elif model_name == "Swin-T":
        return SwinTransformer(num_classes=num_classes, model_name='swin_tiny_patch4_window7_224', 
                              pretrained_path="pretrained_models\swin_tiny_patch4_window7_224.safetensors")
    else:
        raise ValueError(f"未知模型: {model_name}") 