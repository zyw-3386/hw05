#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##代码来源：https://mp.weixin.qq.com/s/iBNvhk-uAeAfTuanxiLs9Q ##

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置中文字体（避免显示方块）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 定义模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, 16 * 14 * 14)
        return self.fc(x)

# ==================== 加载数据 ====================
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True), \
           torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

start_time = time.time()  # 放在训练开始前

# ==================== 训练函数（记录损失） ====================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_losses = []  # 记录每个epoch的损失
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    return train_losses

# ==================== 测试并可视化预测结果 ====================
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()
    
    # 画图
    plt.figure(figsize=(12, 5))
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(2, 5, i+1)
        img = images[i][0].numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print(f"预测结果图已保存为 predictions.png")

# ==================== 主程序 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

train_loader, test_loader = load_data()
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练并记录损失
train_losses = train(model, train_loader, criterion, optimizer, device)

end_time = time.time()
elapsed = end_time - start_time
print(f"总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

# ==================== 图1：训练损失曲线 ====================
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), train_losses, marker='o', linewidth=2)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png', dpi=150)
plt.show()
print(f"训练损失曲线图已保存为 training_loss.png")

# ==================== 图2：预测结果可视化 ====================
visualize_predictions(model, test_loader, device, num_images=10)

# 测试准确率
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'\n测试集准确率: {100 * correct / total:.2f}%')

