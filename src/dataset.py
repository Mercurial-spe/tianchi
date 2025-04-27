import numpy as np
import time
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset

class GalaxyDataset(Dataset):
    """楼道图像数据集类"""
    
    def __init__(self, images, labels, transform=None):
        """初始化数据集
        
        Args:
            images (list): 图像路径列表
            labels (list): 标签列表
            transform: 数据变换
        """
        self.images = images
        self.labels = labels
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        """获取数据项
        
        Args:
            index (int): 索引
            
        Returns:
            tuple: (图像张量, 标签, 图像路径)
        """
        # 记录起始时间，用于计算读取耗时(调试用)
        start_time = time.time()
        
        # 获取图像路径
        image_path = self.images[index]
        
        # 打开并转换图像
        img = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        # 返回图像张量、标签和图像路径
        return img, torch.from_numpy(np.array(self.labels[index])), image_path
    
    def __len__(self):
        """获取数据集长度"""
        return len(self.labels) 