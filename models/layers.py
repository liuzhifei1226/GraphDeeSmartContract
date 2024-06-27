import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from parser import parameter_parser
from torch.nn.parameter import Parameter

args = parameter_parser()


class GraphSAGE(nn.Module):
    def __init__(self, in_features, out_features, activation=None, dropout=0.0, normalize=True):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.normalize = normalize

        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj):
        # x: node features [N, in_features]
        # adj: adjacency matrix [N, N]

        # Aggregate neighbor features
        neighbor_features = torch.spmm(adj, x)  # [N, in_features]

        # Concatenate node features with aggregated neighbor features
        combined = torch.cat([x, neighbor_features], dim=1)  # [N, in_features * 2]

        # Apply dropout if specified
        if self.dropout:
            combined = self.dropout(combined)

        # Apply linear transformation
        out = self.linear(combined)  # [N, out_features]

        # Apply activation function if specified
        if self.activation:
            out = self.activation(out)

        # Normalize embeddings if specified
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)

        return out
# GraphConv layers and models
# class GraphConv(nn.Module):
#     """
#     Graph Convolution Layer & Additional tricks (power of adjacency matrix and weighted self connections)
#     n_relations: number of relation types (adjacency matrices)
#     """
#
#     def __init__(self, in_features, out_features, n_relations=1,
#                  activation=None, adj_sq=False, scale_identity=False):
#         super(GraphConv, self).__init__()
#         self.fc = nn.Linear(in_features=in_features * n_relations, out_features=out_features)
#         self.n_relations = n_relations
#         self.activation = activation
#         self.adj_sq = adj_sq
#         self.scale_identity = scale_identity
#
#     def laplacian_batch(self, A):
#         batch, N = A.shape[:2]
#         if self.adj_sq:
#             A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
#         I = torch.eye(N).unsqueeze(0).to(args.device)
#         a = I.shape
#         if self.scale_identity:
#             I = 2 * I  # increase weight of self connections
#         # A_hat = A
#         # add I represents self connections of nodes
#         A_hat = I + A
#         # D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)  # Adjacent matrix normalization
#         # L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
#         L = A_hat  # remove D_hat
#         return L
#
#     def forward(self, data):
#         x, A, mask = data[:3]
#         if len(A.shape) == 3:
#             A = A.unsqueeze(3)
#         x_hat = []
#         for rel in range(self.n_relations):
#             x_hat.append(torch.bmm(self.laplacian_batch(A[:, :, :, rel]), x))
#         x = self.fc(torch.cat(x_hat, 2))
#
#         if len(mask.shape) == 2:
#             mask = mask.unsqueeze(2)
#         x = x * mask
#         # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc
#         # which affects callee_node embeddings in the following layers
#         if self.activation is not None:
#             x = self.activation(x)
#         return x, A, mask

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, n_relations=1, activation=None, adj_sq=False, scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features * n_relations, out_features)
        self.n_relations = n_relations
        self.activation = activation
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity
        self.bn = nn.BatchNorm1d(out_features)  # 添加批归一化层

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        I = torch.eye(N).unsqueeze(0).to(A.device)
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        A_hat = I + A
        D_hat = torch.sum(A_hat, 1) ** (-0.5)
        D_hat = torch.diag_embed(D_hat)
        L = torch.bmm(torch.bmm(D_hat, A_hat), D_hat)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []
        for rel in range(self.n_relations):
            x_hat.append(torch.bmm(self.laplacian_batch(A[:, :, :, rel]), x))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)
        x = x * mask

        if self.activation is not None:
            x = self.activation(x)

        # 调整形状以适应批归一化层
        batch_size, num_nodes, num_features = x.shape
        x = x.view(batch_size * num_nodes, num_features)  # (batch_size * num_nodes, out_features)
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, num_features)  # 恢复原来的形状
        return x, A, mask


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        batch = input.shape[0]
        h_prime = []
        for i in range(batch):
            h = torch.mm(input[i], self.W)
            N = h.size()

            f_1 = torch.matmul(h, self.a1)
            f_2 = torch.matmul(h, self.a2)
            e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj[i] > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            out = torch.matmul(attention, h)
            h_prime.append(out.data.numpy())

        h_prime = torch.from_numpy(np.array(h_prime))

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        batch = input.shape[0]
        output = []
        for i in range(batch):
            if len(input.shape) == 3:
                support = torch.mm(input[i], self.weight)
                out = torch.spmm(adj[i], support)
                output.append(out.data.numpy())
            else:
                support = torch.mm(input[i], self.weight)
                out = torch.spmm(adj[i], support[i])
                output.append(out.data.numpy())
        # output = torch.mean(output, dim=0, keepdim=True)
        output = torch.from_numpy(np.array(output))
        if self.bias is not None:
            output = output + self.bias
            return output
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
