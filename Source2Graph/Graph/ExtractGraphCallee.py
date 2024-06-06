import os
import re
import time
import numpy as np

# 将用户定义的变量映射到符号名称（var）
var_list = ['balances[msg.sender]', 'participated[msg.sender]', 'playerPendingWithdrawals[msg.sender]',
            'nonces[msgSender]', 'balances[beneficiary]', 'transactions[transactionId]', 'tokens[token][msg.sender]',
            'totalDeposited[token]', 'tokens[0][msg.sender]', 'accountBalances[msg.sender]', 'accountBalances[_to]',
            'creditedPoints[msg.sender]', 'balances[from]', 'withdrawalCount[from]', 'balances[recipient]',
            'investors[_to]', 'Bal[msg.sender]', 'Accounts[msg.sender]', 'Holders[_addr]', 'balances[_pd]',
            'ExtractDepositTime[msg.sender]', 'Bids[msg.sender]', 'participated[msg.sender]', 'deposited[_participant]',
            'Transactions[TransHash]', 'm_txs[_h]', 'balances[investor]', 'this.balance', 'proposals[_proposalID]',
            'accountBalances[accountAddress]', 'Chargers[id]', 'latestSeriesForUser[msg.sender]',
            'balanceOf[_addressToRefund]', 'tokenManage[token_]', 'milestones[_idMilestone]', 'payments[msg.sender]',
            'rewardsForA[recipient]', 'userBalance[msg.sender]', 'credit[msg.sender]', 'credit[to]', 'round_[_rd]',
            'userPendingWithdrawals[msg.sender]', '[msg.sender]', '[from]', '[to]', '[_to]', "msg.sender"]

# 函数限制类型
function_limit = ['private', 'onlyOwner', 'internal', 'onlyGovernor', 'onlyCommittee', 'onlyAdmin', 'onlyPlayers',
                  'onlyManager', 'onlyHuman', 'only_owner', 'onlyCongressMembers', 'preventReentry', 'onlyMembers',
                  'onlyProxyOwner', 'ownerExists', 'noReentrancy', 'notExecuted', 'noReentrancy', 'noEther',
                  'notConfirmed']

# bool条件表达式:
var_op_bool = ['!', '~', '**', '*', '!=', '<', '>', '<=', '>=', '==', '<<', '>>', '||', '&&']

# 赋值表达式
var_op_assign = ['|=', '=', '^=', '&=', '<<=', '>>=', '+=', '-=', '*=', '/=', '%=', '++', '--']


# 切分合约的所有函数
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    flag = -1
    flag1 = 0

    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" and len(function_list) > 0:
                flag1 = 0
        if flag1 == 0:
            if len(text) > 0 and text != "\n":
                if text.split()[0] == "function" or text.split()[0] == "function()":
                    function_list.append([text])
                    flag += 1
                elif len(function_list) > 0 and ("function" in function_list[flag][0]):
                    if text.split()[0] != "modifier" and text.split()[0] != "event":
                        function_list[flag].append(text)
                    else:
                        flag1 += 1
                        continue
        else:
            continue

    return function_list


# 定位 call.value 生成图
def generate_graph(filepath):
    allFunctionList = split_function(filepath)  # 存储所有函数
    print("===allFunctionList===:\n", allFunctionList)
    callValueList = []  # 存储所有调用了 call.value 的W函数
    cFunctionList = []  # 存储一个 调用了W函数的C 函数
    CFunctionLists = []  # 存储所有调用了W函数的C 函数
    withdrawNameList = []  # 存储调用了 call.value的W函数的函数名
    otherFunctionList = []  # 存储W函数以外的函数
    node_list = []  # 存储所有节点
    edge_list = []  # 存储所有边及其特征
    node_feature_list = []  # 存储所有节点特征
    params = []  # 存储W函数的参数
    param = []
    key_count = 0  # S和W节点的数量
    c_count = 0  # C节点数量

    edgeDataDepend = []  # 存储涉及数据依赖的边上面的数据

    # ---------------------------  处理节点  ----------------------------

    # 存储除了W的其他函数
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])
    print("===otherFunctionList====:\n", otherFunctionList)

    # 遍历所有函数, 找到 call.value 关键字, 存储S和W节点
    for i in range(len(allFunctionList)):
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                node_list.append("S")
                node_list.append("W" + str(key_count))
                callValueList.append([allFunctionList[i], "S", "W" + str(key_count)])

                # 获取函数名和参数
                ss = allFunctionList[i][0]
                pp = re.compile(r'[(](.*?)[)]', re.S)
                result = re.findall(pp, ss)
                print("===result_params====:\n", result)
                if len(result) != 0:
                    result_params = result[0].split(",")

                    for n in range(len(result_params)):
                        param.append(result_params[n].strip().split(" ")[-1])

                    params.append([param, "S", "W" + str(key_count)])
                    print("===params====:\n", params)
                # 处理W函数权限限制, 用作权限属性
                # 默认有C节点
                limit_count = 0
                for k in range(len(function_limit)):
                    if function_limit[k] in callValueList[key_count][0][0]:
                        limit_count += 1
                        if "address" in text:
                            node_feature_list.append(
                                ["S", "LimitedAC", ["W" + str(key_count)],
                                 2, "INNADD", 0])
                            node_feature_list.append(
                                ["W" + str(key_count), "LimitedAC", [],
                                 1, "NULL", 1])
                            # 添加边的数据依赖  数据
                            # edgeDataDepend.append(["S", "W" + str(key_count), "address"])
                            break
                        elif "msg.sender" in text:
                            node_feature_list.append(
                                ["S", "LimitedAC", ["W" + str(key_count)],
                                 2, "MSG", 0])
                            node_feature_list.append(
                                ["W" + str(key_count), "LimitedAC", [],
                                 1, "NULL", 1])
                            # 添加边的数据依赖  数据
                            # edgeDataDepend.append(["S", "W" + str(key_count), "msg.sender"])
                            break
                        else:
                            param_count = 0
                            for pa in param:
                                if pa in text and pa != "":
                                    param_count += 1
                                    node_feature_list.append(
                                        ["S", "LimitedAC",
                                         ["W" + str(key_count)],
                                         2, "MSG", 0])
                                    node_feature_list.append(
                                        ["W" + str(key_count), "LimitedAC", [],
                                         1, "NULL", 1])
                                    # 添加边的数据依赖  数据
                                    # edgeDataDepend.append(["S", "W" + str(key_count), "msg.sender"])
                                    break
                            if param_count == 0:
                                node_feature_list.append(
                                    ["S", "LimitedAC", ["W" + str(key_count)],
                                     2, "INNADD", 0])
                                node_feature_list.append(
                                    ["W" + str(key_count), "LimitedAC", [],
                                     1, "NULL", 0])
                            break

                if limit_count == 0:
                    if "address" in text:
                        node_feature_list.append(
                            ["S", "NoLimit", ["W" + str(key_count)],
                             2, "INNADD", 0])
                        node_feature_list.append(
                            ["W" + str(key_count), "NoLimit", [],
                             1, "NULL", 1])
                        # 添加边的数据依赖  数据
                        # edgeDataDepend.append(["S", "W" + str(key_count), "address"])
                    elif "msg.sender" in text:
                        node_feature_list.append(
                            ["S", "NoLimit", ["W" + str(key_count)],
                             2, "MSG", 0])
                        node_feature_list.append(
                            ["W" + str(key_count), "NoLimit", [],
                             1, "NULL", 1])
                        # 添加边的数据依赖  数据
                        # edgeDataDepend.append(["S", "W" + str(key_count), "msg.sender"])
                    else:
                        param_count = 0
                        for pa in param:
                            if pa in text and pa != "":
                                param_count += 1
                                node_feature_list.append(
                                    ["S", "NoLimit", ["W" + str(key_count)],
                                     2, "MSG", 0])
                                node_feature_list.append(
                                    ["W" + str(key_count), "NoLimit", [],
                                     1, "NULL", 1])
                                # 添加边的数据依赖  数据
                                # edgeDataDepend.append(["S", "W" + str(key_count), "msg.sender"])
                                break
                        if param_count == 0:
                            node_feature_list.append(
                                ["S", "NoLimit", ["W" + str(key_count)],
                                 2, "INNADD", 0])
                            node_feature_list.append(
                                ["W" + str(key_count), "NoLimit", [],
                                 1, "NULL", 1])
                            # 添加边的数据依赖  数据
                            # edgeDataDepend.append(["S", "W" + str(key_count), "address"])

                # function transfer(address _to, uint _value, bytes _data, string _custom_fallback)
                # 获取函数名： (transfer)
                tmp = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))')
                result_withdraw = tmp.findall(allFunctionList[i][0])
                withdrawNameTmp = result_withdraw[1]
                if withdrawNameTmp == "payable":
                    withdrawName = withdrawNameTmp
                else:
                    '''
                    函数名后加左括号
                    '''
                    withdrawName = withdrawNameTmp + "("
                withdrawNameList.append(["W" + str(key_count), withdrawName])
                print("===withdrawNameList====:\n", withdrawNameList)
                key_count += 1
    # 第一个节点表示状态节点（State Node），使用标识符 "S" 表示，没有限制（NoLimit），不包含任何依赖（["NULL"]），权重为0，没有备注。
    # 第二个节点表示提款函数（Withdrawal Function），使用标识符 "W" 表示，没有限制（NoLimit），不包含任何依赖（["NULL"]），权重为0，没有备注。
    # 第三个节点表示调用函数（Calling Function），使用标识符 "C" 表示，没有限制（NoLimit），不包含任何依赖（["NULL"]），权重为0，没有备注。
    # C：调用函数节点
    if key_count == 0:
        print("不存在关键字: call.value,添加默认节点")
        node_feature_list.append(["S", "NoLimit", ["NULL"], 0, "NULL", 0])
        node_feature_list.append(["W0", "NoLimit", ["NULL"], 0, "NULL", 0])
        node_feature_list.append(["C0", "NoLimit", ["NULL"], 0, "NULL", 0])
    else:
        print("遍历所有函数并找到调用W函数的C函数节点")
        # 遍历所有函数并找到调用W函数的C函数节点
        # 通过匹配参数的数量来确定函数调用
        for k in range(len(withdrawNameList)):
            w_key = withdrawNameList[k][0]
            w_name = withdrawNameList[k][1]
            for i in range(len(otherFunctionList)):
                if len(otherFunctionList[i]) > 2:
                    for j in range(1, len(otherFunctionList[i])):
                        text = otherFunctionList[i][j]
                        if w_name in text:
                            p = re.compile(r'[(](.*?)[)]', re.S)
                            result = re.findall(p, text)

                            if len(result) != 0:
                                result_params = result[0].split(",")
                            else:
                                break
                            try:
                                if result_params[0] != "" and len(result_params) == len(params[k][0]):
                                    cFunctionList += otherFunctionList[i]
                                    CFunctionLists.append(
                                        [w_key, w_name, "C" + str(c_count), otherFunctionList[i]])
                                    node_list.append("C" + str(c_count))

                                    for n in range(len(node_feature_list)):
                                        if w_key in node_feature_list[n][0]:
                                            node_feature_list[n][2].append("C" + str(c_count))

                                    # 处理 C 函数 权限限制
                                    limit_count = 0
                                    for m in range(len(function_limit)):
                                        if function_limit[m] in cFunctionList[0]:
                                            limit_count += 1
                                            node_feature_list.append(
                                                ["C" + str(c_count), "LimitedAC", ["NULL"], 0, "NULL", -1])
                                            break
                                    if limit_count == 0:
                                        node_feature_list.append(
                                            ["C" + str(c_count), "NoLimit", ["NULL"], 0, "NULL", -1])
                                    c_count += 1
                                    break
                            except Exception as e:
                                print(f"Error: {e}")

        if c_count == 0:
            print("没有C节点，添加默认节点")
            node_list.append("C0")
            node_feature_list.append(["C0", "NoLimit", ["NULL"], 0, "NULL", 0])
            for n in range(len(node_feature_list)):
                if "W" in node_feature_list[n][0]:
                    node_feature_list[n][2] = ["NULL"]
        print("===node_feature_list====:\n", node_feature_list)
        # ---------------------------  处理边  ----------------------------

        # (1)处理 W->S (包括: W->VAR, VAR->S, S->VAR)
        for i in range(len(callValueList)):
            flag = 0  # flag: flag = 0, call.value之前; flag > 0, call.value之后
            before_var_count = 0
            after_var_count = 0
            var_tmp = []
            var_name = []
            var_w_name = []
            print("处理边w->s: ===callValueList[i][0]====:\n", callValueList[i][0])
            for j in range(len(callValueList[i][0])):
                text = callValueList[i][0][j]
                if '.call.value' not in text:
                    if flag == 0:
                        print("call.value 前\n")
                        # 处理 W -> VAR
                        print("===处理 W -> VAR====\n")
                        for k in range(len(var_list)):
                            if var_list[k] in text:
                                node_list.append("VAR" + str(before_var_count))
                                var_tmp.append("VAR" + str(before_var_count))

                                if len(var_w_name) == 0:
                                    if "assert" in text:
                                        edge_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'AH'])
                                    elif "require" in text:
                                        edge_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'RG'])
                                    elif j >= 1:
                                        if "if" in callValueList[i][0][j - 1]:
                                            edge_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'GN'])
                                        elif "for" in callValueList[i][0][j - 1]:
                                            edge_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'FOR'])
                                        elif "else" in callValueList[i][0][j - 1]:
                                            edge_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'GB'])
                                        elif j + 1 < len(callValueList[i][0]):
                                            if "if" and "throw" in callValueList[i][0][j] or "if" in \
                                                    callValueList[i][0][j] \
                                                    and "throw" in callValueList[i][0][j + 1]:
                                                edge_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'IT'])
                                            elif "if" and "revert" in callValueList[i][0][j] or "if" in \
                                                    callValueList[i][0][
                                                        j] and "revert" in callValueList[i][0][j + 1]:
                                                edge_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'RH'])
                                            elif "if" in text:
                                                edge_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'IF'])
                                            else:
                                                edge_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'FW'])
                                        else:
                                            edge_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'FW'])
                                    else:
                                        edge_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'FW'])

                                    var_node = 0
                                    var_bool_node = 0
                                    for b in range(len(var_op_bool)):
                                        if var_op_bool[b] in text:
                                            node_feature_list.append(
                                                ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1, 'BOOL'])
                                            var_node += 1
                                            var_bool_node += 1
                                            break

                                    if var_bool_node == 0:
                                        for a in range(len(var_op_assign)):
                                            if var_op_assign[a] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'ASSIGN'])
                                                var_node += 1
                                                break

                                    if var_node == 0:
                                        node_feature_list.append(
                                            ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                             callValueList[i][2], 1, 'NULL'])

                                    var_w_name.append(var_list[k])
                                    var_name.append(var_list[k])
                                    before_var_count += 1
                                    print("处理边w->s: ===var_w_name====:\n", var_w_name)
                                    print("处理边w->s: ===var_name====:\n", var_name)
                                else:
                                    var_w_count = 0
                                    for n in range(len(var_w_name)):
                                        if var_list[k] == var_w_name[n]:
                                            var_w_count += 1
                                            var_tmp.append(var_tmp[len(var_tmp) - 1])

                                            var_node = 0
                                            var_bool_node = 0
                                            for b in range(len(var_op_bool)):
                                                if var_op_bool[b] in text:
                                                    node_feature_list.append(
                                                        [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                         callValueList[i][2], 1, 'BOOL'])
                                                    var_bool_node += 1
                                                    var_node += 1
                                                    break

                                            if var_bool_node == 0:
                                                for a in range(len(var_op_assign)):
                                                    if var_op_assign[a] in text:
                                                        node_feature_list.append(
                                                            [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                             callValueList[i][2], 1, 'ASSIGN'])
                                                        var_node += 1
                                                        break

                                            if var_node == 0:
                                                node_feature_list.append(
                                                    [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][2], 1, 'NULL'])

                                    if var_w_count == 0:
                                        var_node = 0
                                        var_bool_node = 0
                                        var_tmp.append("VAR" + str(before_var_count))

                                        for b in range(len(var_op_bool)):
                                            if var_op_bool[b] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'BOOL'])
                                                var_node += 1
                                                var_bool_node += 1
                                                break

                                        if var_bool_node == 0:
                                            for a in range(len(var_op_assign)):
                                                if var_op_assign[a] in text:
                                                    node_feature_list.append(
                                                        ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                         callValueList[i][2], 1, 'ASSIGN'])
                                                    var_node += 1
                                                    break

                                        if var_node == 0:
                                            node_feature_list.append(
                                                ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1, 'NULL'])

                    elif flag != 0:
                        print("call.value 后")
                        # 处理 S->VAR
                        print("===处理 S -> VAR====\n")
                        var_count = 0
                        for k in range(len(var_list)):
                            if var_list[k] in text:
                                if before_var_count == 0:
                                    node_list.append("VAR" + str(after_var_count))
                                    var_tmp.append("VAR" + str(after_var_count))

                                    if "assert" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'AH'])
                                    elif "require" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RG'])
                                    elif "return" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RE'])
                                    elif "if" and "throw" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'IT'])
                                    elif "if" and "revert" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RH'])
                                    elif "if" in text:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'IF'])
                                    else:
                                        edge_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'FW'])

                                    var_node = 0
                                    var_bool_node = 0
                                    for b in range(len(var_op_bool)):
                                        if var_op_bool[b] in text:
                                            node_feature_list.append(
                                                ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                                 callValueList[i][1], 3, 'BOOL'])
                                            var_node += 1
                                            var_bool_node += 1
                                            break

                                    if var_bool_node == 0:
                                        for a in range(len(var_op_assign)):
                                            if var_op_assign[a] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                                     callValueList[i][1], 3, 'ASSIGN'])
                                                var_node += 1
                                                break

                                    if var_node == 0:
                                        node_feature_list.append(
                                            ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                             callValueList[i][1], 3, 'NULL'])

                                    # after_var_count += 1

                                elif before_var_count > 0:
                                    for n in range(len(var_name)):
                                        if var_list[k] == var_name[n]:
                                            var_count += 1
                                            if "assert" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'AH'])
                                            elif "require" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RG'])
                                            elif "return" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RE'])
                                            elif "if" and "throw" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'IT'])
                                            elif "if" and "revert" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RH'])
                                            elif "if" in text:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'IF'])
                                            else:
                                                edge_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'FW'])

                                            after_var_count += 1
                    # print("======edge_list====",edge_list)
                    # print("======edge_feature====", edge_feature)
                elif '.call.value' in text:
                    flag += 1

                    if len(var_tmp) > 0:
                        if "assert" in text:
                            edge_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'AH'])
                        elif "require" in text:
                            edge_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RG'])
                        elif "return" in text:
                            edge_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RE'])
                        elif j > 1:
                            if "if" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'GN'])
                            elif "for" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FOR'])
                            elif "else" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'GB'])
                            elif j + 1 < len(callValueList[i][0]):
                                if "if" and "throw" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "throw" in callValueList[i][0][j + 1]:
                                    edge_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'IT'])
                                elif "if" and "revert" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "revert" in callValueList[i][0][j + 1]:
                                    edge_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RH'])
                                elif "if" in text:
                                    edge_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'IF'])
                                else:
                                    edge_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])
                            else:
                                edge_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])
                        else:
                            edge_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])

                    elif len(var_tmp) == 0:
                        if "assert" in text:
                            edge_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'AH'])
                        elif "require" in text:
                            edge_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RG'])
                        elif "return" in text:
                            edge_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RE'])
                        elif j > 1:
                            if "if" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'GN'])
                            elif "for" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FOR'])
                            elif "else" in callValueList[i][0][j - 1]:
                                edge_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'GB'])
                            elif j + 1 < len(callValueList[i][0]):
                                if "if" and "throw" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "throw" in callValueList[i][0][j + 1]:
                                    edge_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'IT'])
                                elif "if" and "revert" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "revert" in callValueList[i][0][j + 1]:
                                    edge_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RH'])
                                elif "if" in text:
                                    edge_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'IF'])
                                else:
                                    edge_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])
                            else:
                                edge_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])
                        else:
                            edge_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])
                    # print("======edge_list2====", edge_list)
                    # print("======edge_feature2====", edge_feature)
        # (2) 处理 C->W (包括 C->VAR, VAR->W)
        print("处理C->W:\n")
        print("C->W:=====CFunctionLists=======\n", CFunctionLists)
        for i in range(len(CFunctionLists)):
            for j in range(len(CFunctionLists[i][3])):
                text = CFunctionLists[i][3][j]
                var_flag = 0
                for k in range(len(var_list)):
                    if var_list[k] in text:
                        var_flag += 1

                        var_node = 0
                        var_bool_node = 0
                        for b in range(len(var_op_bool)):
                            if var_op_bool[b] in text:
                                node_feature_list.append(
                                    ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                     CFunctionLists[i][2], 1, 'BOOL'])
                                var_node += 1
                                var_bool_node += 1
                                break

                        if var_bool_node == 0:
                            for a in range(len(var_op_assign)):
                                if var_op_assign[a] in text:
                                    node_feature_list.append(
                                        ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                         CFunctionLists[i][2], 1, 'ASSIGN'])
                                    var_node += 1
                                    break

                        if var_node == 0:
                            node_feature_list.append(
                                ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                 CFunctionLists[i][2], 1, 'NULL'])

                        if "assert" in text:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'AH'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        elif "require" in text:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'RG'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        elif "if" and "throw" in text:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'IT'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        elif "if" and "revert" in text:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'RH'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        elif "if" in text:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'IF'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        else:
                            edge_list.append(
                                [CFunctionLists[i][2], "VAR" + str(len(var_tmp)), CFunctionLists[i][2], 1, 'FW'])
                            edge_list.append(
                                ["VAR" + str(len(var_tmp)), CFunctionLists[i][0], CFunctionLists[i][2], 2, 'FW'])
                        break

                if var_flag == 0:
                    if "assert" in text:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'AH'])
                    elif "require" in text:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'RG'])
                    elif "if" and "throw" in text:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'IT'])
                    elif "if" and "revert" in text:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'RH'])
                    elif "if" in text:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'IF'])
                    else:
                        edge_list.append(
                            [CFunctionLists[i][2], CFunctionLists[i][0], CFunctionLists[i][2], 1, 'FW'])
                    break
                else:
                    print("C函数不调用相应的W函数")

    # 去掉重复节点
    edge_list = list(set([tuple(t) for t in edge_list]))
    edge_list = [list(v) for v in edge_list]
    node_feature_list_new = []
    [node_feature_list_new.append(i) for i in node_feature_list if not i in node_feature_list_new]
    # node_feature_list = list(set([tuple(t) for t in node_feature_list]))
    # node_feature_list = [list(v) for v in node_feature_list]
    # node_list = list(set(node_list))

    return node_feature_list_new, edge_list


def printResult(file, node_feature, edge_feature):
    main_point = ['S', 'W0', 'W1', 'W2', 'W3', 'W4', 'C0', 'C1', 'C2', 'C3', 'C4']
    print("print result: =====node_feature=======\n", node_feature)
    for i in range(len(node_feature)):
        if node_feature[i][0] in main_point:
            tmp = ""
            for j in range(0, len(node_feature[i][2]), 2):
                if j + 1 < len(node_feature[i][2]):
                    tmp += node_feature[i][2][j] + "," + node_feature[i][2][j + 1] + " "
                elif len(node_feature[i][2]) == 1:
                    tmp = node_feature[i][2][j]

            node_feature[i][2] = tmp.strip()

    # 获取当前脚本文件的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造相对路径基于脚本文件所在目录
    nodeOutPath = "../graph_data/reentrancy/callee_node/" + file
    edgeOutPath = "../graph_data/reentrancy/callee_edge/" + file
    node_full_path = os.path.join(script_dir, nodeOutPath)
    edge_full_path = os.path.join(script_dir, edgeOutPath)

    f_node = open(node_full_path, 'a')
    for feature in node_feature:
        result = " ".join(map(str, feature))
        f_node.write(result + '\n')
    f_node.close()

    f_edge = open(edge_full_path, 'a')
    for edge in edge_feature:
        result = " ".join(map(str, edge))
        f_edge.write(result + '\n')
    f_edge.close()

    return node_feature, edge_feature


def getGraph(filepath):
    contract_path = filepath
    node_feature, edge_feature = generate_graph(contract_path)
    node_feature = sorted(node_feature, key=lambda x: (x[0]))
    edge_feature = sorted(edge_feature, key=lambda x: (x[2], x[3]))
    # node_feature, edge_feature = generate_potential_fallback_node(node_feature, edge_feature)
    print("node_feature", node_feature)
    print("edge_feature", edge_feature)
    printResult(contract_path.split("/")[4], node_feature, edge_feature)


if __name__ == "__main__":
    test_contract = "../source_code/SimpleDAO.sol"
    node_feature, edge_feature = generate_graph(test_contract)
    node_feature = sorted(node_feature, key=lambda x: (x[0]))
    edge_feature = sorted(edge_feature, key=lambda x: (x[2], x[3]))
    # node_feature, edge_feature = generate_potential_fallback_node(node_feature, edge_feature)
    print("node_feature", node_feature)
    print("edge_feature", edge_feature)
    printResult(test_contract.split("/")[2], node_feature, edge_feature)
