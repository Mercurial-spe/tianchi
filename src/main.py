import os
import wandb

from src.config import *
from src.data_loader import (
    load_train_data, load_test_data, 
    get_train_val_split, get_train_loader, 
    get_val_loader, get_test_loader
)
from src.models import get_model1, get_model3
from src.trainer import ModelTrainer
from src.predictor import ModelPredictor

def main():
    """主函数，执行完整的训练和预测流程"""
    
    # 初始化W&B（可选）
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
        
    # 数据准备
    print("\n===== 数据准备 =====")
    train_df = load_train_data()
    train_idx, val_idx = get_train_val_split(train_df)
    
    # 创建数据加载器
    print("\n===== 创建数据加载器 =====")
    train_loader = get_train_loader(train_df, train_idx)
    val_loader = get_val_loader(train_df, val_idx)
    
    # 创建保存模型的目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练ResNet18模型
    print("\n===== 训练ResNet18模型 =====")
    resnet_trainer = ModelTrainer(
        model_name="ResNet18",
        model=get_model1(),
        train_loader=train_loader,
        val_loader=val_loader
    )
    resnet_trainer.train()
    
    # 训练EfficientNet-B0模型
    print("\n===== 训练EfficientNet-B0模型 =====")
    efficientnet_trainer = ModelTrainer(
        model_name="EfficientNet-B0",
        model=get_model3(),
        train_loader=train_loader,
        val_loader=val_loader
    )
    efficientnet_trainer.train()
    
    # 测试数据处理和预测
    print("\n===== 测试预测和提交 =====")
    test_df = load_test_data()
    test_loader = get_test_loader(test_df)
    
    # 使用最佳模型预测（选择EfficientNet-B0作为最终模型）
    predictor = ModelPredictor(get_model3(), test_loader)
    predictor.load_best_model("EfficientNet-B0")
    predictor.predict_and_submit()
    
    if USE_WANDB:
        wandb.finish()
    
    print("\n===== 流程完成 =====")

if __name__ == "__main__":
    main()