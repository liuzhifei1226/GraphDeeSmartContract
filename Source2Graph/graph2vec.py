import os
import json
import numpy as np
from Source2Graph.vec2onehot import vec2onehot

"""
S, W, C nips_features: Node nips_features + Edge nips_features + Var nips_features;
节点自身属性 + 入变量 + 出变量 + 入边 + 出边

将特定的属性名称转换为整数编码
"""

dict_AC = {"NULL": 0, "LimitedAC": 1, "NoLimit": 2}

dict_NodeName = {"NULL": 0, "VAR0": 1, "VAR1": 2, "VAR2": 3, "VAR3": 4, "VAR4": 5, "VAR5": 6, "S": 7, "W0": 8,
                 "W1": 9, "W2": 10, "W3": 11, "W4": 12, "C0": 13, "C1": 14, "C2": 15, "C3": 16, "C4": 17}

dict_VarOpName = {"NULL": 0, "BOOL": 1, "ASSIGN": 2}

dict_EdgeOpName = {"NULL": 0, "FW": 1, "IF": 2, "GB": 3, "GN": 4, "WHILE": 5, "FOR": 6, "RE": 7, "AH": 8, "RG": 9,
                   "RH": 10, "IT": 11}

dict_AllOpName = {"NULL": 0, "FW": 1, "ASSIGN": 2, "BOOL": 3, "IF": 4, "GB": 5, "GN": 6, "WHILE": 7, "FOR": 8, "RE": 9,
                  "AH": 10, "RG": 11, "RH": 12, "IT": 13}

dict_NodeOpName = {"NULL": 0, "MSG": 1, "INNADD": 2}

dict_ConName = {"NULL": 0, "ARG1": 1, "ARG2": 2, "ARG3": 3, "CON1": 4, "CON2": 5, "CON3": 6, "CNS1": 7, "CNS2": 8,
                "CNS3": 9}

node_convert = {"S": 0, "W0": 1, "C0": 2, "W1": 3, "C1": 4, "W2": 5, "C2": 6, "W3": 7, "C3": 8, "W4": 9, "C4": 10,
                "VAR0": 0, "VAR1": 1, "VAR2": "VAR2", "VAR3": "VAR3", "VAR4": "VAR4", "VAR5": "VAR5"}

v2o = vec2onehot()  # 创建one-hot字典


# 从输入文件中提取每个节点的属性 #
def extract_node_features(nodeFile):
    nodeNum = 0
    node_list = []
    node_attribute_list = []

    f = open(nodeFile)
    lines = f.readlines()
    f.close()

    for line in lines:
        node = list(map(str, line.split()))
        verExist = False
        for i in range(0, len(node_list)):
            if node[1] == node_list[i]:
                verExist = True
            else:
                continue
        if verExist is False:
            node_list.append(node[1])
            nodeNum += 1
        node_attribute_list.append(node)

    return nodeNum, node_list, node_attribute_list


# 消除子图中的冗余节点
def elimination_node(node_attribute_list):
    print("=========node_attribute_list before elimination======:", node_attribute_list)
    main_point = ['S', 'W0', 'W1', 'W2', 'W3', 'W4', 'C0', 'C1', 'C2', 'C3', 'C4']
    extra_var_list = []  # 提取低优先级var
    for i in range(0, len(node_attribute_list)):
        if node_attribute_list[i][1] not in main_point:
            if i + 1 < len(node_attribute_list):
                if node_attribute_list[i][1] == node_attribute_list[i + 1][1]:
                    loc1 = int(node_attribute_list[i][3])  # 相对位置
                    op1 = node_attribute_list[i][4]  # 运算
                    loc2 = int(node_attribute_list[i + 1][3])
                    op2 = node_attribute_list[i + 1][4]
                    if loc2 - loc1 == 1:
                        op1_index = dict_VarOpName[op1]
                        op2_index = dict_VarOpName[op2]
                        # 基于优先级提取被调用节点属性
                        if op1_index < op2_index:
                            extra_var_list.append(node_attribute_list.pop(i))
                        else:
                            extra_var_list.append(node_attribute_list.pop(i + 1))
    print("=========node_attribute_list after elimination======:", node_attribute_list)
    return node_attribute_list, extra_var_list


# 将节点的属性进行嵌入编码
def embedding_node(node_attribute_list):
    # 消除后嵌入每个被调用合约的节点 #
    node_encode = []
    var_encode = []
    node_embedding = []
    var_embedding = []
    main_point = ['S', 'W0', 'W1', 'W2', 'W3', 'W4', 'C0', 'C1', 'C2', 'C3', 'C4']

    for j in range(0, len(node_attribute_list)):
        v = node_attribute_list[j][0]
        print("====node_attribute_list[j]======", node_attribute_list[j])
        if v in main_point:
            vf0 = node_attribute_list[j][0]
            # vf1 = dict_NodeName[node_attribute_list[j][1]]
            # vfm1 = v2o.node2vecEmbedding(node_attribute_list[j][1])
            vf1 = dict_AC[node_attribute_list[j][1]]
            vfm1 = v2o.nodeAC2vecEmbedding(node_attribute_list[j][1])

            result = node_attribute_list[j][2].split(",")
            for call_vec in range(len(result)):
                if call_vec + 1 < len(result):
                    tmp_vf = str(dict_NodeName[result[call_vec]]) + "," + str(dict_NodeName[result[call_vec + 1]])
                    tmp_vfm = np.array(list(v2o.node2vecEmbedding(result[call_vec]))) ^ np.array(
                        list(v2o.node2vecEmbedding(result[call_vec + 1])))
                elif len(result) == 1:
                    tmp_vf = dict_NodeName[result[call_vec]]
                    tmp_vfm = v2o.node2vecEmbedding(result[call_vec])
            vf2 = tmp_vf
            vfm2 = tmp_vfm
            vf3 = int(node_attribute_list[j][3])
            vfm3 = v2o.sn2vecEmbedding(node_attribute_list[j][3])
            vf4 = dict_NodeOpName[node_attribute_list[j][4]]
            vfm4 = v2o.nodeOP2vecEmbedding(node_attribute_list[j][4])
            # 新增  依赖关系向量转化
            vf5 = int(node_attribute_list[j][5])
            vfm5 = v2o.depend2vecEmbedding(node_attribute_list[j][5])

            nodeEmbedding = vfm1.tolist() + vfm2.tolist() + vfm3.tolist() + vfm4.tolist() + vfm5.tolist()
            node_embedding.append([vf0, np.array(nodeEmbedding)])
            temp = [vf1, vf2, vf3, vf4, vf5]
            node_encode.append([vf0, temp])
        else:
            vf0 = node_attribute_list[j][0]
            vf1 = dict_NodeName[node_attribute_list[j][1]]
            vfm1 = v2o.node2vecEmbedding(node_attribute_list[j][1])
            vf2 = dict_NodeName[node_attribute_list[j][2]]
            vfm2 = v2o.node2vecEmbedding(node_attribute_list[j][2])
            vf3 = int(node_attribute_list[j][3])
            vfm3 = v2o.sn2vecEmbedding(node_attribute_list[j][3])
            vf4 = dict_VarOpName[node_attribute_list[j][4]]
            vfm4 = v2o.varOP2vecEmbedding(node_attribute_list[j][4])
            vf5 = int(dict_NodeOpName['NULL'])
            vfm5 = v2o.nodeOP2vecEmbedding('NULL')
            varEmbedding = vfm1.tolist() + vfm2.tolist() + vfm3.tolist() + vfm4.tolist() + vfm5.tolist()
            var_embedding.append([vf0, np.array(varEmbedding)])
            temp = [vf1, vf2, vf3, vf4, vf5]
            var_encode.append([vf0, temp])

    return node_encode, var_encode, node_embedding, var_embedding


# 消除多余的边
def elimination_edge(edgeFile):
    # 消除被调用合约的多余边 #
    edge_list = []  # 所有边
    extra_edge_list = []  # 被消除的边

    f = open(edgeFile)
    lines = f.readlines()
    f.close()

    for line in lines:
        edge = list(map(str, line.split()))
        edge_list.append(edge)

    # 消融两个节点之间的多个边
    for k in range(0, len(edge_list)):
        if k + 1 < len(edge_list):
            start1 = edge_list[k][0]  # 开始节点
            end1 = edge_list[k][1]  # 结束节点
            op1 = edge_list[k][4]
            start2 = edge_list[k + 1][0]
            end2 = edge_list[k + 1][1]
            op2 = edge_list[k + 1][4]
            if start1 == start2 and end1 == end2:
                op1_index = dict_EdgeOpName[op1]
                op2_index = dict_EdgeOpName[op2]
                # extract callee_edge attribute based on priority
                if op1_index < op2_index:
                    extra_edge_list.append(edge_list.pop(k))
                else:
                    extra_edge_list.append(edge_list.pop(k + 1))

    return edge_list, extra_edge_list


# 对边进行属性嵌入编码
def embedding_edge(edge_list):
    edge_encode = []
    edge_embedding = []

    for k in range(len(edge_list)):
        start = edge_list[k][0]  # 开始节点
        end = edge_list[k][1]  # 结束节点
        a, b, c = edge_list[k][2], edge_list[k][3], edge_list[k][4]  # 原始信息

        ef1 = dict_NodeName[a]
        ef2 = int(b)
        ef3 = dict_EdgeOpName[c]

        ef_temp = [ef1, ef2, ef3]
        edge_encode.append([start, end, ef_temp])

        efm1 = v2o.node2vecEmbedding(a)
        efm2 = v2o.sn2vecEmbedding(b)
        efm3 = v2o.edgeOP2vecEmbedding(c)

        efm_temp = efm1.tolist() + efm2.tolist() + efm3.tolist()
        edge_embedding.append([start, end, np.array(efm_temp)])

    return edge_encode, edge_embedding


# 构建节点和边的向量表示，结合节点的自属性、入变量、出变量、入边、出边：
def construct_vec(edge_list, node_embedding, var_embedding, edge_embedding, edge_encode):
    print("开始构建被调用合约图节点向量...")
    var_in_node = []
    var_in = []
    var_out_node = []
    var_out = []
    edge_in_node = []
    edge_in = []
    edge_out_node = []
    edge_out = []
    node_vec = []
    F_point = ['F']
    S_point = ['S']
    W_point = ['W0', 'W1', 'W2', 'W3', 'W4']
    C_point = ['C0', 'C1', 'C2', 'C3', 'C4']
    main_point = ['S', 'W0', 'W1', 'W2', 'W3', 'W4', 'C0', 'C1', 'C2', 'C3', 'C4']
    node_embedding_dim_without_edge = 250

    if len(var_embedding) > 0:
        for k in range(len(edge_embedding)):
            if edge_list[k][0] in F_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][1]):
                        var_out.append([edge_embedding[k][0], var_embedding[i][1]])
                        edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in F_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][0]):
                        var_in.append([edge_embedding[k][1], var_embedding[i][1]])
                        edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            if edge_list[k][0] in C_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][1]):
                        var_out.append([edge_embedding[k][0], var_embedding[i][1]])
                        edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in C_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][0]):
                        var_in.append([edge_embedding[k][1], var_embedding[i][1]])
                        edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            elif edge_list[k][0] in W_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][1]):
                        var_out.append([edge_embedding[k][0], var_embedding[i][1]])
                        edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
                        break
            elif edge_list[k][1] in W_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][0]):
                        var_in.append([edge_embedding[k][1], var_embedding[i][1]])
                        edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            elif edge_list[k][0] in S_point:
                S_OUT = []
                S_OUT_Flag = 0
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][1]):
                        S_OUT.append(var_embedding[i][1])
                        S_OUT_Flag = 1
                if S_OUT_Flag != 1:
                    S_OUT.append(np.zeros(len(var_embedding[0][1]), dtype=int))
                var_out.append([edge_embedding[k][0], S_OUT[0]])
                edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in S_point:
                for i in range(len(var_embedding)):
                    if str(var_embedding[i][0]) == str(edge_embedding[k][0]):
                        var_in.append([edge_embedding[k][1], var_embedding[i][1]])
                        edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])
                        break
            else:
                print("Edge from callee_node %s to callee_node %s:  edgeFeature: %s" % (
                    edge_embedding[k][0], edge_embedding[k][1], edge_embedding[k][2]))
    else:
        for k in range(len(edge_embedding)):
            if edge_list[k][0] in F_point:
                edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in F_point:
                edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            if edge_list[k][0] in C_point:
                edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in C_point:
                edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            elif edge_list[k][0] in W_point:
                edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in W_point:
                edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

            elif edge_list[k][0] in S_point:
                edge_out.append([edge_embedding[k][0], edge_embedding[k][2]])
            elif edge_list[k][1] in S_point:
                edge_in.append([edge_embedding[k][1], edge_embedding[k][2]])

    edge_vec_length = 44
    var_vec_length = 61

    for i in range(len(var_in)):
        var_in_node.append(var_in[i][0])
    for i in range(len(var_out)):
        var_out_node.append(var_out[i][0])
    for i in range(len(edge_in)):
        edge_in_node.append(edge_in[i][0])
    for i in range(len(edge_out)):
        edge_out_node.append(edge_out[i][0])

    for i in range(len(main_point)):
        if main_point[i] not in var_in_node:
            var_in.append([main_point[i], np.zeros(var_vec_length, dtype=int)])
        if main_point[i] not in var_out_node:
            var_out.append([main_point[i], np.zeros(var_vec_length, dtype=int)])
        if main_point[i] not in edge_out_node:
            edge_out.append([main_point[i], np.zeros(edge_vec_length, dtype=int)])
        if main_point[i] not in edge_in_node:
            edge_in.append([main_point[i], np.zeros(edge_vec_length, dtype=int)])

    varIn_dict = dict(var_in)
    varOut_dict = dict(var_out)
    edgeIn_dict = dict(edge_in)
    edgeOut_dict = dict(edge_out)

    for i in range(len(node_embedding)):
        vec = np.zeros(node_embedding_dim_without_edge, dtype=int)
        if node_embedding[i][0] in F_point:
            node_feature = node_embedding[i][1].tolist() + np.array(varIn_dict[node_embedding[i][0]]).tolist() + \
                           np.array(varOut_dict[node_embedding[i][0]]).tolist()
            vec[0:len(np.array(node_feature))] = np.array(node_feature)
            node_vec.append([node_embedding[i][0], vec])
        elif node_embedding[i][0] in S_point:
            node_feature = node_embedding[i][1].tolist() + np.array(varIn_dict[node_embedding[i][0]]).tolist() + \
                           np.array(varOut_dict[node_embedding[i][0]]).tolist()
            vec[0:len(np.array(node_feature))] = np.array(node_feature)
            node_vec.append([node_embedding[i][0], vec])
        elif node_embedding[i][0] in W_point:
            node_feature = node_embedding[i][1].tolist() + np.array(varIn_dict[node_embedding[i][0]]).tolist() + \
                           np.array(varOut_dict[node_embedding[i][0]]).tolist()
            vec[0:len(np.array(node_feature))] = np.array(node_feature)
            node_vec.append([node_embedding[i][0], vec])
        elif node_embedding[i][0] in C_point:
            node_feature = node_embedding[i][1].tolist() + np.array(varIn_dict[node_embedding[i][0]]).tolist() + \
                           np.array(varOut_dict[node_embedding[i][0]]).tolist()
            vec[0:len(np.array(node_feature))] = np.array(node_feature)
            node_vec.append([node_embedding[i][0], vec])

    for i in range(len(node_vec)):
        node_vec[i][1] = node_vec[i][1].tolist()

    print("节点 vec:")
    for i in range(len(node_vec)):
        node_vec[i][0] = node_convert[node_vec[i][0]]
        print(node_vec[i][0], node_vec[i][1])

    for i in range(len(edge_embedding)):
        edge_embedding[i][2] = edge_embedding[i][2].tolist()

    # "S" -> 0, W0 -> 1, C0 -> 2
    if len(edge_encode) == 2:
        end = edge_encode[len(edge_encode) - 2][1]
        start = edge_encode[len(edge_encode) - 1][0]
        flag = edge_encode[len(edge_encode) - 1][1]
        if end == start and ('VAR' in flag or 'MSG' in flag):
            edge_encode[len(edge_encode) - 1][1] = edge_encode[len(edge_encode) - 2][0]

    if len(edge_encode) > 2:
        end1 = edge_encode[len(edge_encode) - 1][1]
        start2 = edge_encode[len(edge_encode) - 2][0]
        if end1 == start2 and ('VAR' in end1 or 'MSG' in end1):
            edge_encode[len(edge_encode) - 1][1] = edge_encode[len(edge_encode) - 3][0]

    for i in range(len(edge_encode)):
        if i + 1 < len(edge_encode):
            start1 = edge_encode[i][0]
            end1 = edge_encode[i][1]
            start2 = edge_encode[i + 1][0]

            if end1 == start2 and ('VAR' in end1 or 'MSG' in end1):
                edge_encode[i][1] = edge_encode[i + 1][1]
                edge_encode[i + 1][0] = edge_encode[i][0]
            elif 'W' in start1 and 'VAR' in end1:
                edge_encode[i][1] = 'S'

    print("边 Vec:")
    for i in range(len(edge_encode)):
        edge_encode[i][0] = node_convert[edge_encode[i][0]]
        edge_encode[i][1] = node_convert[edge_encode[i][1]]
        print(edge_encode[i][0], edge_encode[i][1], edge_encode[i][2])

    graph_edge = []

    for i in range(len(edge_encode)):
        graph_edge.append([edge_encode[i][0], edge_encode[i][2][2], edge_encode[i][1]])
        print(f"edge_encode[i][0]:{edge_encode[i][0]},edge_encode[i][2][2]:{edge_encode[i][2][2]},edge_encode[i][1]:{edge_encode[i][1]}")

    print(graph_edge)

    return node_vec, graph_edge


def getVec(filename):
    # 获取当前脚本文件的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造相对路径基于脚本文件所在目录
    node = "./graph_data/reentrancy/callee_node/" + filename
    edge = "./graph_data/reentrancy/callee_edge/" + filename
    node_full_path = os.path.join(script_dir, node)
    edge_full_path = os.path.join(script_dir, edge)


    nodeNum, node_list, node_attribute_list = extract_node_features(node_full_path)
    node_attribute_list, extra_var_list = elimination_node(node_attribute_list)
    node_encode, var_encode, node_embedding, var_embedding = embedding_node(node_attribute_list)
    edge_list, extra_edge_list = elimination_edge(edge_full_path)
    edge_encode, edge_embedding = embedding_edge(edge_list)
    node_vec, graph_edge = construct_vec(edge_list, node_embedding, var_embedding, edge_embedding, edge_encode)
    return node_vec, graph_edge


if __name__ == "__main__":
    node = "./graph_data/callee_node/SimpleDAO.sol"
    edge = "./graph_data/callee_edge/SimpleDAO.sol"
    nodeNum, node_list, node_attribute_list = extract_node_features(node)
    node_attribute_list, extra_var_list = elimination_node(node_attribute_list)
    node_encode, var_encode, node_embedding, var_embedding = embedding_node(node_attribute_list)
    edge_list, extra_edge_list = elimination_edge(edge)
    edge_encode, edge_embedding = embedding_edge(edge_list)
    node_vec, graph_edge = construct_vec(edge_list, node_embedding, var_embedding, edge_embedding, edge_encode)
