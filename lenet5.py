#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##代码仿照公众号文章：https://mp.weixin.qq.com/s/iBNvhk-uAeAfTuanxiLs9Q ##

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 定义 LeNet-5 模型 ====================
class LeNet5(nn.Module):
    """
    适配 MNIST (28x28) 的 LeNet-5 变体
    经典 LeNet-5 输入是 32x32，这里调整了第一层卷积的padding
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 卷积层 输入1通道(灰度) -> 输出6通道, 卷积核5x5, padding=2 保持28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # S2: 池化层 2x2 步长2 -> 28x28 -> 14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 卷积层 6 -> 16通道, 卷积核5x5 (无padding -> 14x14 -> 10x10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # S4: 池化层 2x2 步长2 -> 10x10 -> 5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 卷积层 16 -> 120通道, 卷积核5x5 (正好把5x5卷成1x1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # F6: 全连接层 120 -> 84
        self.fc1 = nn.Linear(120, 84)
        # 输出层 84 -> 10 (数字0-9)
        self.fc2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # C1 -> ReLU -> S2
        x = self.pool1(self.relu(self.conv1(x)))   # 28 -> 14
        # C3 -> ReLU -> S4
        x = self.pool2(self.relu(self.conv2(x)))   # 14 -> 5
        # C5 -> ReLU (这里不需要池化，因为卷积直接输出1x1)
        x = self.relu(self.conv3(x))               # 5 -> 1
        # 展平
        x = x.view(-1, 120)
        # F6 -> ReLU
        x = self.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# ==================== 2. 数据加载（同任务一） ====================
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

# ==================== 3. 训练函数 ====================
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    train_losses = []
    for epoch in range(epochs):
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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    return train_losses

# ==================== 4. 测试函数 ====================
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

# ==================== 5. 可视化预测结果（可选） ====================
def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    images = images.cpu()
    
    plt.figure(figsize=(12, 5))
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(2, 5, i+1)
        img = images[i][0].numpy()
        plt.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('lenet5_predictions.png', dpi=150)
    plt.show()

# ==================== 6. 主程序 ====================
def main():
    # 计时开始
    start_time = time.time()
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    batch_size = 64
    train_loader, test_loader = load_data(batch_size=batch_size)
    
    # 创建模型
    model = LeNet5().to(device)
    
    # 打印模型结构（可选）
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    epochs = 5
    print(f'\n开始训练 {epochs} 轮...')
    train_losses = train(model, train_loader, criterion, optimizer, device, epochs)
    
    # 测试
    print('\n测试结果:')
    test_loss, test_accuracy = test(model, test_loader, criterion, device)
    
    # 计时结束
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'\n总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)')
    
    # 保存模型
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print('模型已保存为 lenet5_mnist.pth')
    
    # 可选：画训练损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), train_losses, marker='o', linewidth=2)
    plt.title('LeNet-5 Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('lenet5_training_loss.png', dpi=150)
    plt.show()
    
    # 可选：可视化预测结果
    visualize_predictions(model, test_loader, device, num_images=10)
    
    return test_accuracy

# 运行
if __name__ == '__main__':
    acc = main()

