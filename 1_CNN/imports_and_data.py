import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 设置matplotlib在Jupyter notebook中显示图像的命令在脚本中不需要

# 数据预处理
transform = transforms.ToTensor()

# 加载数据集
train_data = datasets.MNIST(root='dataset/cnn_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='dataset/cnn_data', train=False, download=True, transform=transform)
