import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset, DataLoader


def create_dataset():
    # coef := 斜率
    # y = coef * x + intercept + noise
    x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True)
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x, y, coef

# 构建数据集
x, y, coef = create_dataset()

# 构造数据集对象
dataset = TensorDataset(x, y)

# 构造数据集加载器
# dataset =: 数据集对象
# batch_size =: 批量训练样本数据
# shuffle =: 样本数据是否打乱
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 构造模型
# in_features =: 输入张量的大小size
# out_features =: 输出张量的大小size
model = torch.nn.Linear(in_features=1, out_features=1)

# 模型训练
# 损失函数
criterion = torch.nn.MSELoss()
# 优化函数
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for _ in range(epochs):
    loss_sum = 0
    samples = 0
    for x, y in dataloader:
        # 将一个batch的训练数据数据输入模型
        y_pred = model(x.type(torch.float32))
        # 计算损失
        loss = criterion(y_pred, y.reshape(-1, 1).type(torch.float32))
        loss_sum += loss.item()
        samples += len(x)

        # 梯度清零
        # By default, PyTorch accumulates (adds up) gradients instead of replacing them
        optimizer.zero_grad()
        # 自动微分（反向传播）
        loss.backward()
        # 更新参数
        optimizer.step()

    losses.append(loss_sum / samples)

plt.plot(range(epochs), losses)
plt.title("loss vs epochs")
plt.grid()
plt.show(block=True)

x = torch.linspace(x.min(), x.max(), 1000)
y1 = torch.tensor([v * model.weight + model.bias for v in x])
y2 = torch.tensor([v * coef + 1.5 for v in x])
plt.plot(x, y1, label="model")
plt.plot(x, y2, label="true")
plt.grid()
plt.legend()    
plt.show(block=True)