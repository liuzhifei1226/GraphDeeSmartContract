# 使用Solidity AST解析源代码
from solcx import compile_standard
from solcx import install_solc

install_solc("0.8.0")


# 解析Solidity源代码并获取AST
def parse_source_code(source_code):
    compiled_sol = compile_standard({
        "language": "Solidity",
        "sources": {
            "contract.sol": {
                "content": source_code
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["*"]
                }
            }
        }
    })
    return compiled_sol


# 构建图形数据结构的节点和边
def build_graph(ast):
    nodes = []
    edges = []

    # 检查AST结构
    print("======ast======\n ", ast)
    print("ast[contracts][contract.sol]", ast["contracts"]["contract.sol"])
    # 遍历AST并构建节点和边
    for source_unit in ast["contracts"]["contract.sol"]:
        for node_type, node_data in source_unit.items():
            if node_type == "ContractDefinition":
                contract_name = node_data["name"]
                # 创建合约节点
                contract_node = {
                    "type": "Contract",
                    "name": contract_name,
                    # 其他属性
                }
                nodes.append(contract_node)

                # 分析函数和变量
                for member in node_data["nodes"]:
                    if member["nodeType"] == "FunctionDefinition":
                        # 创建函数节点
                        function_name = member["name"]
                        function_node = {
                            "type": "Function",
                            "name": function_name,
                            # 其他属性
                        }
                        nodes.append(function_node)

                        # 创建函数与合约之间的边
                        edge = {
                            "start_node": contract_name,
                            "end_node": function_name,
                            # 其他属性
                        }
                        edges.append(edge)

                    elif member["nodeType"] == "VariableDeclaration":
                        # 创建变量节点
                        variable_name = member["name"]
                        variable_node = {
                            "type": "Variable",
                            "name": variable_name,
                            # 其他属性
                        }
                        nodes.append(variable_node)

                        # 创建变量与函数或合约之间的边
                        # ...

                    # 添加其他节点和边的处理逻辑

    return nodes, edges


# 主函数
def main():

    install_solc("latest")  # 安装最新版本的Solidity编译器

    source_code = """
    // Solidity源代码示例
    contract MyContract {
        uint256 public myVariable;

        function myFunction() public {
            myVariable = 123;
        }
    }
    """
    ast = parse_source_code(source_code)
    nodes, edges = build_graph(ast)
    # 处理节点和边的数据结构
    # 可选择可视化或存储图形数据结构


if __name__ == "__main__":
    main()
