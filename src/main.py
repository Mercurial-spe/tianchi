import os
import wandb

from src.config import *
from src.data_loader import (
    load_train_data, load_test_data, 
    get_train_val_split, get_train_loader, 
    get_val_loader, get_test_loader
)
from src.models import get_model
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
    
    # 创建保存模型的目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 训练模型列表
    # models_to_train = ["ResNet18", "EfficientNet-B0"]
    models_to_train = []
    # 是否启用高级模型
    use_advanced_models = True
    if use_advanced_models:
        models_to_train.extend(["Swin-T"])
    
    best_acc = 0.0
    best_model_name = ""
    
    # 训练所有模型
    for model_name in models_to_train:
        print(f"\n===== 训练{model_name}模型 =====")
        
        # 为当前模型创建特定的数据加载器
        print(f"\n===== 为{model_name}创建数据加载器 =====")
        # 使用高级数据增强和类别平衡采样器，传入模型名称以使用正确的图像尺寸
        train_loader = get_train_loader(
            train_df, 
            train_idx, 
            use_rand_augment=USE_RAND_AUGMENT,
            use_balanced_sampler=USE_BALANCED_SAMPLER,
            model_name=model_name
        )
        val_loader = get_val_loader(train_df, val_idx, model_name=model_name)
        
        # 获取模型实例
        model = get_model(model_name)
        
        # 创建训练器
        trainer = ModelTrainer(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            use_mixup=USE_MIXUP,
            mixup_alpha=MIXUP_ALPHA,
            scheduler_type=SCHEDULER_TYPE
        )
        
        # 训练模型
        acc = trainer.train()
        
        # 记录最佳模型
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
    
    print(f"\n===== 所有模型训练完成，最佳模型：{best_model_name}，准确率：{best_acc:.4f} =====")
    
    # 测试数据处理和预测
    print("\n===== 测试预测和提交 =====")
    test_df = load_test_data()
    test_loader = get_test_loader(test_df, model_name=best_model_name)
    
    # 使用最佳模型预测
    model = get_model(best_model_name)
    predictor = ModelPredictor(model, test_loader)
    predictor.load_best_model(best_model_name)
    predictor.predict_and_submit()
    
    if USE_WANDB:
        wandb.finish()
    
    print("\n===== 流程完成 =====")

if __name__ == "__main__":
    main()