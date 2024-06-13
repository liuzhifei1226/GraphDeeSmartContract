def generate_pairs(start, end):
    edges = []
    for i in range(start, end, 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges
# 指定生成数据的范围
start_number = 1
end_number = 18  # 这里设置为最终生成的最大数加1

generate_pairs(start_number, end_number)






# import os
# from os.path import join as pjoin
# import numpy as np
# import Source2Graph.graph2vec
#
# """
# 读取训练数据集的所有txt文件
# """
#
#
# def parse_txt_file(fpath, line_parse_fn=None):
#     with open(pjoin(data_dir, fpath), 'r') as f:
#         lines = f.readlines()
#     data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
#     return data
#
#
# # 读取图的邻接矩阵,  A
# def read_graph_adj(fpath, nodes, graphs):
#     edges = parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
#     adj_dict = {}
#     for edge in edges:
#         node1 = int(edge[0].strip()) - 1
#         node2 = int(edge[1].strip()) - 1
#         graph_id = nodes[node1]
#         print(f"node1: {node1}, node2: {node2}, graph_id : {graph_id}, nodes[node1] : {nodes[node1]}, nodes[node2] : {nodes[node2]}")
#         assert graph_id == nodes[node2], ('invalid training_data', graph_id, nodes[node2])
#         if graph_id not in adj_dict:
#             n = len(graphs[graph_id])
#             adj_dict[graph_id] = np.zeros((n, n))
#         ind1 = np.where(graphs[graph_id] == node1)[0]
#         ind2 = np.where(graphs[graph_id] == node2)[0]
#         assert len(ind1) == len(ind2) == 1, (ind1, ind2)
#         adj_dict[graph_id][ind1, ind2] = 1
#     adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
#     return adj_list
#
#
# # 读取图的节点关系,  graph_indicator
# def read_graph_nodes_relations(fpath):
#     graph_ids = parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
#     nodes, graphs = {}, {}
#     for node_id, graph_id in enumerate(graph_ids):
#         print(f"node_id : {node_id}, graph_id: {graph_id}")
#         if graph_id not in graphs:
#             graphs[graph_id] = []
#         graphs[graph_id].append(node_id)
#         nodes[node_id] = graph_id
#     graph_ids = np.unique(list(graphs.keys()))
#     unique_id = graph_ids
#     for graph_id in graph_ids:
#         graphs[graph_id] = np.array(graphs[graph_id])
#     print(f"nodes : {nodes}, graphs: {graphs},  unique_id: {unique_id}")
#     return nodes, graphs, unique_id
#
#
# # 读取节点特征,node_labels
# def read_node_features(fpath, nodes, graphs, fn):
#     node_features_all = parse_txt_file(fpath, line_parse_fn=fn)
#     node_features = {}
#     for node_id, x in enumerate(node_features_all):
#         graph_id = nodes[node_id]
#         if graph_id not in node_features:
#             node_features[graph_id] = [None] * len(graphs[graph_id])
#         ind = np.where(graphs[graph_id] == node_id)[0]
#         assert len(ind) == 1, ind
#         assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
#         node_features[graph_id][ind[0]] = x
#     node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
#     return node_features_lst


# data = {}
# rnd_state = None
# use_cont_node_attr = False
# data_dir = '../training_data/REENTRANCY_CORENODES_1671'
# files = os.listdir(data_dir)
# nodes, graphs, unique_id = read_graph_nodes_relations(
#     list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
# data['features'] = read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
#                                       nodes, graphs, fn=lambda s: int(s.strip()))
# data['adj_list'] = read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
# data['targets'] = np.array(
#     parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
#                    line_parse_fn=lambda s: int(float(s.strip()))))
# data['ids'] = unique_id
# if use_cont_node_attr:
#     data['attr'] = read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
#                                       nodes, graphs,
#                                       fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

# node_vec,graph_edge = Source2Graph.graph2vec.getVec("0x0a3fba29c8941fb09f6c712c06d2eade82df225b.sol")
# print(f"node_vec: {node_vec}")
# print(f"graph_edge: {graph_edge}")