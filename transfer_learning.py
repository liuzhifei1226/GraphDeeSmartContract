import numpy as np
import torch
import torch.nn as nn
from SMVulDetector import DataReader
from models.gcn_modify import GCN_MODIFY
from torch.utils.data import DataLoader
from load_data import GraphData
from load_data import split_ids, GraphData, collate_batch


import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, GCN_MODIFY.parameters()), lr=0.001)
seed = 50
rnd_state = np.random.RandomState(seed)
datareader = DataReader(data_dir='./training_data/LOOP_FULLNODES_1317/', rnd_state=rnd_state,
                        use_cont_node_attr=False, folds=1)


loaders = []
for split in ['train', 'test']:
    gdata = GraphData(fold_id=1, datareader=datareader, split=1)
    loader = DataLoader(gdata, batch_size=128, shuffle=split.find('train') >= 0,
                            num_workers=2, collate_fn=collate_batch)
    loaders.append(loader)


GCN_MODIFY(in_features=loaders[0].dataset.num_features,
                           out_features=loaders[0].dataset.num_classes,
                           n_hidden=256,
                           filters='64,64,64',
                           dropout=0.3,
                           adj_sq=True,
                           scale_identity='store_true').to('cpu')
GCN_MODIFY.load_state_dict(torch.load('FFG.pth'))

# 冻结图卷积层的参数
for param in GCN_MODIFY.gconv.parameters():
    param.requires_grad = False

# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    GCN_MODIFY.train()
    for data in loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = GCN_MODIFY(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

GCN_MODIFY.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in loaders[1]:
        inputs, labels = data
        outputs = GCN_MODIFY(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
