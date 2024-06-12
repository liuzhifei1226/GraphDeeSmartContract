def generate_sparse_adjacency_matrix(num_blocks, block_size):
    edges = []
    for b in range(num_blocks):
        start = b * block_size + 1
        for i in range(start, start + block_size - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

    return edges

def write_edges_to_file(edges, filename):
    with open(filename, 'w') as f:
        for edge in edges:
            f.write(f"{edge[0]}, {edge[1]}\n")

# 设置块的数量和每个块的大小
num_blocks = 50  # 50 blocks for 100 nodes, each block has 2 nodes
block_size = 2

# 生成边列表
edges = generate_sparse_adjacency_matrix(num_blocks, block_size)

print(edges)

# import numpy as np
# import scipy.sparse as sp
# '''
# create_adjacency_matrix 函数：接收边列表和节点数，返回一个稀疏邻接矩阵。
# block_diagonal_adjacency 函数：将多个邻接矩阵组合成一个块对角线的稀疏矩阵。
# edges_list 和 num_nodes_list：分别定义了每个图的边列表和节点数。
# 使用 block_diag 函数将多个图的邻接矩阵组合成一个块对角线的邻接矩阵。
#
# 生成图的邻接矩阵
# '''
# def create_adjacency_matrix(edges, num_nodes):
#     """Creates a sparse adjacency matrix from edges."""
#     row, col = zip(*edges)
#     data = np.ones(len(edges))
#     adjacency_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
#     # Since the graph is undirected, make the matrix symmetric
#     adjacency_matrix = adjacency_matrix + adjacency_matrix.T - sp.diags(adjacency_matrix.diagonal())
#     return adjacency_matrix
#
# def block_diagonal_adjacency(matrices):
#     """Combines multiple adjacency matrices into a block diagonal sparse matrix."""
#     return sp.block_diag(matrices, format='coo')
#
# # Example graphs
# edges_list = [
#     [(0, 1), (1, 2), (2, 0)],  # Graph 1
#     [(0, 1), (1, 2)],          # Graph 2
#     [(0, 1), (1, 3), (3, 4)]   # Graph 3
# ]
#
# num_nodes_list = [3, 3, 5]
#
# adjacency_matrices = [create_adjacency_matrix(edges, num_nodes) for edges, num_nodes in zip(edges_list, num_nodes_list)]
#
# block_diagonal_matrix = block_diagonal_adjacency(adjacency_matrices)
#
# # Print the block diagonal adjacency matrix
# print(block_diagonal_matrix)
# print(block_diagonal_matrix.toarray())
