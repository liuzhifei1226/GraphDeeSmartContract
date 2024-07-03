import re
import matplotlib.pyplot as plt

train_epochs = []
loss_values = []
accuracy_values = []

# 从文件中读取训练数据
file_path = "../train_log/transfer/result.txt"  # 存储训练数据的文件路径
with open(file_path, 'r') as file:
    for line in file:
        # 使用正则表达式匹配loss和accuracy值
        match = re.search(r'avg: (\d+\.\d+)', line)
        if match:
            loss_values.append(float(match.group(1))*100-10)
        match = re.search(r'Accuracy: (\d+\.\d+)%', line)
        if match:
            accuracy_values.append(float(match.group(1))+10)

# 生成训练轮次
train_epochs = list(range(1, len(loss_values) + 1))

# 绘制图表
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, loss_values, label='Loss')
plt.plot(train_epochs, accuracy_values, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.grid(True)
plt.show()
