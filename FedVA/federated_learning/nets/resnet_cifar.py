import torch.nn as nn
from torchvision.models import resnet18

def ResNet18():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove initial max pool for CIFAR10
    return model
