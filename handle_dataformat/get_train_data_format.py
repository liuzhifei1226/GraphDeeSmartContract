import Source2Graph.Graph.ExtractGraphCallee
import Source2Graph.graph2vec
import os
import numpy as np
import Source2Graph
import handle_dataformat.train_data_A

novul = ["C0 NoLimit NULL 0 NULL 0\n", "S NoLimit NULL 0 NULL 0\n", "W0 NoLimit NULL 0 NULL 0\n"]


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
        file_path = directory_path + f
        if not os.path.isfile("../Source2Graph/graph_data/reentrancy/callee_node" + f):
            Source2Graph.Graph.ExtractGraphCallee.getGraph(file_path)
        else:
            continue


def source2traindata(graphdata_path):
    node_path = graphdata_path + "callee_node/"
    edge_path = graphdata_path + "callee_edge/"
    training_path = "../training_data/REENTRANCY_CORENODES_1671/"
    # 文件初始化
    A = []
    graph_indicator = []
    graph_labels = []
    node_labels = []
    node_attributes = []

    for root, dirs, files in os.walk(node_path):
        graph_id = 1
        node_count = 0
        num = 0
        for file in files:


            file_path = os.path.join(root, file)
            try:
                # 生成图的节点和边向量
                node_vec, graph_edge = Source2Graph.graph2vec.getVec(file)
            except Exception as e:
                print(f"转换向量失败 {file_path}：{e}")
            with open(file_path, 'r', encoding='utf-8') as f:
                print("num:", num)
                num += 1
                lines = f.readlines()

                line_count = len(lines)
                if line_count == 0:
                    continue

                node_count += line_count
                # 插入图标签

                if lines == novul:
                    graph_labels.append([0])
                else:
                    graph_labels.append([1])

                for i in range(0, line_count):
                    # 插入 图 indicator
                    graph_indicator.append([graph_id])
                    # 插入节点标签
                    node_labels.append([1])

                # 插入节点 attributes 向量
                for vec in node_vec:
                    str_list = [str(x) for x in list(map(float, vec[1]))]

                    # 使用逗号连接字符串列表
                    joined_str = ', '.join(str_list)
                    node_attributes.append(joined_str)

                # print(f"node_vec: {node_vec}")
                # print(f"graph_edge: {graph_edge}")



            graph_id += 1
        # 生成并插入图邻接矩阵
        matric = handle_dataformat.train_data_A.generate_pairs(1, node_count)
        for i in matric:
            str_list = [str(x) for x in i]

            # 使用逗号连接字符串列表
            joined_str = ', '.join(str_list)
            A.append(joined_str)
        # print(f"A: {A}")
        # print(f"graph_labels: {graph_labels}")
        # print(f"graph_indicator: {graph_indicator}")
        # print(f"node_labels: {node_labels}")
        # print(f"node_attributes: {node_attributes}")

    # 保存文件
    np.savetxt(training_path + 'DS_A.txt', A, fmt='%s')
    np.savetxt(training_path + 'DS_graph_indicator.txt', graph_indicator, fmt='%d')
    np.savetxt(training_path + 'DS_graph_labels.txt', graph_labels, fmt='%d')
    np.savetxt(training_path + 'DS_node_labels.txt', node_labels, fmt='%d')
    np.savetxt(training_path + 'DS_node_attributes.txt', node_attributes, fmt='%s')


if __name__ == '__main__':
    dataset_path = "../dataset/vul_kinds_marked/reentrancy/"
    graphdata_path = "../Source2Graph/graph_data/reentrancy/"
    files = list_files_in_directory(dataset_path)
    # print("文件列表:", files)
    # source2graph(dataset_path, files)
    source2traindata(graphdata_path)
