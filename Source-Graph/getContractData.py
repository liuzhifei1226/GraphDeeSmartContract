from solcx import compile_standard, install_solc,get_installable_solc_versions
import json


versions = get_installable_solc_versions()
print(versions)  # 打印可用的solc版本列表
install_solc("0.8.9")  # 安装指定版本的solc编译器


# Solidity智能合约源代码
contract_source_code = """
pragma solidity ^0.8.0;

contract MyContract {
    uint256 public number;

    constructor(uint256 _number) {
        number = _number;
    }

    function addNumber(uint256 _value) public {
        number += _value;
    }

    function subtractNumber(uint256 _value) public {
        number -= _value;
    }
}
"""

# 编译Solidity合约并获取AST
compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {"MyContract.sol": {"content": contract_source_code}},
        "settings": {"outputSelection": {"*": {"*": ["*"]}}},
    }
)

contract_name = list(compiled_sol["contracts"]["MyContract.sol"].keys())[0]
contract_json = compiled_sol["contracts"]["MyContract.sol"][contract_name]
print(contract_json)

ast = json.loads(contract_json["evm"]["deployedBytecode"]["sourceMap"])

# 遍历AST并存储每个函数的变量
functions_variables = {}

for node in ast["children"]:
    if "name" in node and node["name"] == contract_name:
        for func in node["children"]:
            if "name" in func:
                function_name = func["name"]
                variables = []
                for var in func["variables"]:
                    variables.append(var["name"])
                functions_variables[function_name] = variables

print(functions_variables)
