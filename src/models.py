import torch.nn as nn
import torchvision.models as models

def get_model1():
    model = models.resnet18(True)
    model.fc = nn.Linear(512, 5)
    return model

def get_model3():
    model = models.efficientnet_b0(True)
    model.classifier = nn.Linear(1280, 5)
    return model 