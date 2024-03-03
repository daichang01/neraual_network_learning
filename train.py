# train.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import Model
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn

# 设置随机种子以确保结果可重现
torch.manual_seed(41)

# 加载数据集
my_df = pd.read_csv('dataset/iris.csv')
# 数据预处理
my_df['species'] = my_df['species'].replace(['setosa', 'versicolor', 'virginica'], [0.0, 1.0, 2.0])
X = my_df.drop('species', axis=1).values
y = my_df['species'].values

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# 转换为Tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 实例化模型
model = Model()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    if i % 10 == 0:
        print(f'Epoch {i} loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 绘制损失曲线
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
