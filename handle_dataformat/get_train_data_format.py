import Source2Graph.Graph.ExtractGraphCallee
import os
import numpy as np


def list_files_in_directory(directory_path):
    try:
        # 获取目录中的所有文件和文件夹
        entries = os.listdir(directory_path)

        # 仅获取文件
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]

        return files
    except Exception as e:
        print(f"Error: {e}")
        return []


def source2graph(directory_path, filelist):
    for f in filelist:
        file_path = directory_path + '/' + f
        if not os.path.isfile("../Source2Graph/graph_data/reentrancy/callee_node"+ f):
            Source2Graph.Graph.ExtractGraphCallee.getGraph(file_path)
        else:
            continue

def source2vec(graphdata_path):
    node_path = graphdata_path + "callee_node"
    edge_path = graphdata_path + "callee_edge"

    # 构造数据
    graphs = [
        {
            'nodes': [0, 1, 2],
            'edges': [(0, 1), (1, 2)],
            'label': 1,
            'node_labels': [1, 2, 3],
            'edge_labels': [10, 20],
            'node_attributes': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            'edge_attributes': [[0.5], [1.5]],
            'graph_attributes': 1.5
        },
        {
            'nodes': [0, 1],
            'edges': [(0, 1)],
            'label': 2,
            'node_labels': [4, 5],
            'edge_labels': [30],
            'node_attributes': [[7.0, 8.0], [9.0, 10.0]],
            'edge_attributes': [[2.5]],
            'graph_attributes': 2.5
        }
    ]

    file_line_counts = {}
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    file_line_counts[file_path] = line_count
            except Exception as e:
                print(f"无法读取文件 {file_path}：{e}")
    return file_line_counts

    # 文件初始化
    edges = []
    graph_indicator = []
    graph_labels = []
    node_labels = []
    edge_labels = []
    node_attributes = []
    edge_attributes = []
    graph_attributes = []

    node_id = 1
    for graph_id, graph in enumerate(graphs, start=1):
        # 处理图的标签
        graph_labels.append(graph['label'])
        graph_attributes.append(graph.get('graph_attributes', ''))

        for node in graph['nodes']:
            # 处理节点的图指示符
            graph_indicator.append(graph_id)
            # 处理节点的标签
            if 'node_labels' in graph:
                node_labels.append(graph['node_labels'][node])
            # 处理节点的属性
            if 'node_attributes' in graph:
                node_attributes.append(','.join(map(str, graph['node_attributes'][node])))

        for edge, (src, dst) in enumerate(graph['edges']):
            # 处理边
            edges.append((node_id + src, node_id + dst))
            edges.append((node_id + dst, node_id + src))
            # 处理边的标签
            if 'edge_labels' in graph:
                edge_labels.append(graph['edge_labels'][edge])
                edge_labels.append(graph['edge_labels'][edge])
            # 处理边的属性
            if 'edge_attributes' in graph:
                edge_attributes.append(','.join(map(str, graph['edge_attributes'][edge])))
                edge_attributes.append(','.join(map(str, graph['edge_attributes'][edge])))

        node_id += len(graph['nodes'])

    # 保存文件
    np.savetxt('DS_A.txt', edges, fmt='%d')
    np.savetxt('DS_graph_indicator.txt', graph_indicator, fmt='%d')
    np.savetxt('DS_graph_labels.txt', graph_labels, fmt='%d')

    if node_labels:
        np.savetxt('DS_node_labels.txt', node_labels, fmt='%d')

    if edge_labels:
        np.savetxt('DS_edge_labels.txt', edge_labels, fmt='%d')

    if node_attributes:
        np.savetxt('DS_node_attributes.txt', node_attributes, fmt='%s')

    if edge_attributes:
        np.savetxt('DS_edge_attributes.txt', edge_attributes, fmt='%s')

    if graph_attributes:
        np.savetxt('DS_graph_attributes.txt', graph_attributes, fmt='%s')


if __name__ == '__main__':
    dataset_path = "../dataset/vul_kinds_marked/reentrancy-pluto-newmark"
    graphdata_path = "../Source2Graph/graph_data/reentrancy/"
    files = list_files_in_directory(dataset_path)
    # print("文件列表:", files)
    # source2graph(directory_path, files)
    source2vec(graphdata_path)
