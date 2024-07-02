import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from SMVulDetector_main import DataReader
from models.gcn_modify import GCN_MODIFY
from torch.utils.data import DataLoader
from load_data import split_ids, GraphData, collate_batch
import torch.optim as optim
print("Running script:", __file__)

print("111111111111111111111111111111111111111111")
seed = 50
n_folds=3
rnd_state = np.random.RandomState(seed)
datareader = DataReader(data_dir='./training_data/LOOP_FULLNODES_1317/', rnd_state=rnd_state,
                        use_cont_node_attr=False, folds=n_folds)
print("Number of graphs in datareader:", datareader.N_graphs)

for fold_id in range(3):
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id, datareader=datareader, split=split)
        print(f"gdata:{gdata.__len__()}")
        print(f"Number of samples in dataset: {len(gdata)}")
        loader = DataLoader(gdata, batch_size=128, shuffle=split.find('train') >= 0,
                            num_workers=2, collate_fn=collate_batch)
        loaders.append(loader)
    # Check number of batches
    for i, loader in enumerate(loaders):
        print(f"Fold {fold_id} Loader {i} - Number of batches: {len(loader)}")


model = GCN_MODIFY(in_features=loaders[0].dataset.num_features,
                       out_features=loaders[0].dataset.num_classes,
                       n_hidden=256,
                       filters=[64,64,64],
                       dropout=0.3,
                       adj_sq=False,
                       scale_identity=False).to('cpu')
ckpt = torch.load('FFG.pth')
ckpt.pop('gconv.0.fc.weight')
msg = model.load_state_dict(ckpt,strict=False)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 冻结图卷积层的参数
# for param in GCN_MODIFY.gconv.parameters():
#     param.requires_grad = False

print("startstartstartstartstartstartstartstartstart")

loss_fn = F.cross_entropy
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,verbose=True)

# 修改后的训练函数
def train(train_loader):
    model.train()
    start = time.time()
    train_loss, n_samples, correct_preds = 0, 0, 0

    for batch_idx, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to('cpu')

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data[4])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        preds = output.argmax(dim=1)
        correct_preds += (preds == data[4]).sum().item()

        train_loss += loss.item() * len(output)
        n_samples += len(output)

    avg_loss = train_loss / n_samples
    accuracy = correct_preds / n_samples
    time_iter = time.time() - start

    scheduler.step(train_loss) # 使用损失更新学习率调度器
    torch.save(model.state_dict(), 'FFG.pth')
    print(
        f'Train Epoch: {epoch + 1} [{n_samples}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)] '
        f'Loss: {loss.item():.6f} (avg: {avg_loss:.6f}) Accuracy: {accuracy * 100:.2f}% sec/iter: {time_iter / (batch_idx + 1):.4f}')


# 修改后的测试函数
def test(test_loader):
    model.eval()
    start = time.time()
    test_loss, n_samples = 0, 0
    tn, fp, fn, tp = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to('cpu')

            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = output.max(1, keepdim=True)[1]

            for k in range(len(pred)):
                if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (np.array(data[4][k].cpu()).tolist() == 1):
                    tp += 1
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4][k].cpu()).tolist() == 0):
                    tn += 1
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4][k].cpu()).tolist() == 1):
                    fn += 1
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                        np.array(data[4][k].cpu()).tolist() == 0):
                    fp += 1

    accuracy = (tp + tn) / n_samples
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0

    avg_loss = test_loss / n_samples
    time_iter = time.time() - start

    print(f'Test set (epoch {epoch + 1}): Average loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, '
          f'Recall: {recall * 100:.2f}%, Precision: {precision * 100:.2f}%, F1-Score: {F1 * 100:.2f}%, '
          f'FPR: {FPR * 100:.2f}%, sec/iter: {time_iter / (batch_idx + 1):.4f}')

    return accuracy, recall, precision, F1, FPR

# 进行多个epoch的训练和测试
for epoch in range(200):
    train(loaders[0])
result_folds = []
accuracy, recall, precision, F1, FPR = test(loaders[1])
result_folds.append([accuracy, recall, precision, F1, FPR])
print(result_folds)
acc_list = []
recall_list = []
precision_list = []
F1_list = []
FPR_list = []

for i in range(len(result_folds)):
    acc_list.append(result_folds[i][0])
    recall_list.append(result_folds[i][1])
    precision_list.append(result_folds[i][2])
    F1_list.append(result_folds[i][3])
    FPR_list.append(result_folds[i][4])

print(
    '{}-fold cross validation avg acc (+- std): {}% ({}%), recall (+- std): {}% ({}%), precision (+- std): {}% ({}%), '
    'F1-Score (+- std): {}% ({}%), FPR (+- fpr): {}% ({}%)'.format(
        n_folds, np.mean(acc_list), np.std(acc_list), np.mean(recall_list), np.std(recall_list),
        np.mean(precision_list), np.std(precision_list), np.mean(F1_list), np.std(F1_list), np.mean(FPR_list),
        np.std(FPR_list))
)