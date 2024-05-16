import numpy as np
import re
def generate_graph_caller(contract_caller, contract_callee):
    caller_node = []
    # 读取合约A文件
    with open(contract_caller, 'r') as file:
        contract_caller_code = file.read()

    # 读取合约B文件
    with open(contract_callee, 'r') as file:
        contract_callee_code = file.read()
    # 使用正则表达式搜索合约B中的合约名称
    contract_name_pattern = re.compile(r'\bcontract\s+(\w+)\s*{')

    contract_name_matches = contract_name_pattern.findall(contract_callee_code)
    print("contract_name_matches:", contract_name_matches)

    if contract_name_matches:
        for i in contract_name_matches:

            # 使用正则表达式搜索合约A中调用合约B的语句
            call_pattern = re.compile(r'\b{}\b\s*\.\s*\w+\s*\(.*\)\s*;'.format(re.escape(i)), re.IGNORECASE)

            call_matches = call_pattern.findall(contract_caller_code)

            if call_matches:
                print("在合约A中找到调用合约B的语句：")
                # for match in call_matches:
                print(call_matches)
                caller_node.append(["C"+i, "NoLimit", ["NULL"], 0, ["CALLER"], 0])
                return caller_node
            else:
                print("在合约A中未找到调用合约B的语句。")


def printResult(file, caller_node):
    nodeOutPath = "../graph_data/caller_node/" + file


    f_node = open(nodeOutPath, 'a')
    if caller_node:
        for item in caller_node:
            print(item)
            for i in item:
                result = " ".join(np.array(i))
                f_node.write(result + '\n')
    else:
        print("在caller合约中未找到调用语句")
    f_node.close()





if __name__ == "__main__":
    contract_caller = "../source_code/caller/smartTest.sol"
    contract_callee = "../source_code/callee/smartTest.sol"
    caller_node = generate_graph_caller(contract_caller, contract_callee)
    printResult(contract_caller.split("/")[2], caller_node)
    print("caller_node:", caller_node)
