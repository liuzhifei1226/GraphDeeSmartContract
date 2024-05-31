import torch
import copy
import torch.utils
import numpy as np

'''
计算数据集的总长度n。
计算每个折叠的步长stride。
将数据集按步长划分为多个测试集test_ids。
确保所有测试集加起来覆盖了整个数据集。
根据测试集构造训练集train_ids，每个训练集不包含对应的测试集数据。
返回训练集和测试集的划分。
'''
# split train and test
def split_ids(ids, folds):
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids


'''
初始化函数__init__：根据数据读取器datareader和折叠编号fold_id、数据集划分split来初始化数据。
set_fold函数：根据指定的折叠设置数据集，包括目标标签、节点特征、邻接矩阵等。
__len__函数：返回数据集的样本数量。
__getitem__函数：返回一个样本的数据，包括节点特征、邻接矩阵、图标签和图ID。 
'''
class GraphData(torch.utils.data.Dataset):
    def __init__(self, datareader, fold_id, split):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.num_classes = data['num_classes']
        self.num_features = data['num_features']
        self.idx = data['splits'][fold_id][self.split]
        # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.ids = copy.deepcopy([data['ids'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # convert to torch
        return [torch.from_numpy(self.features_onehot[index]).float(),  # node_features
                torch.from_numpy(self.adj_list[index]).float(),  # adjacency matrix
                int(self.labels[index]),  # graph labels
                int(self.ids[index])  # graph id
                ]


def collate_batch(batch):
    """
    Creates a batch of same size graphs by zero-padding callee_node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: [node_features * batch_size, A * batch_size, label * batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    collate_batch函数用于将一个批次的数据转换为统一大小的张量，主要步骤包括：

    获取批次的大小B和每个图的节点数N_nodes。
    获取节点特征的维度C。
    找出批次中最大节点数N_nodes_max。
    初始化零填充的张量graph_support、邻接矩阵A和节点特征x。
    将每个图的数据填充到对应的张量中。
    将节点数、标签和ID转换为张量并返回。

    """
    B = len(batch)
    N_nodes = [len(batch[b][1]) for b in range(B)]
    C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes))

    graph_support = torch.zeros(B, N_nodes_max)
    A = torch.zeros(B, N_nodes_max, N_nodes_max)
    x = torch.zeros(B, N_nodes_max, C)
    for b in range(B):
        x[b, :N_nodes[b]] = batch[b][0]
        A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        graph_support[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()
    labels = torch.from_numpy(np.array([batch[b][2] for b in range(B)])).long()
    ids = torch.from_numpy(np.array([batch[b][3] for b in range(B)])).long()

    return [x, A, graph_support, N_nodes, labels, ids]
