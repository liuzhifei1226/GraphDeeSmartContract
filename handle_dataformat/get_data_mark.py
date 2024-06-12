# nohup ./smartbugs -t slither -f dataset/xfuzz_contracts_unmark/*.sol --processes 2 --mem-limit 2g --timeout 600 > output.txt 2>&1 &
from enum import Enum
import csv
import shutil
import os
from fuzzywuzzy import process

vul_slither = ['reentrancy_eth', 'reentrancy_no_eth', 'tx_origin', 'reentrancy_benign', 'reentrancy_events',
               'timestamp',
               'reentrancy_unlimited_gas', 'controlled_delegatecall', 'delegatecall_loop']

vul_mythril = ['Integer_Arithmetic_Bugs_SWC_101', 'Delegatecall_to_user_supplied_address_SWC_112',
               'Dependence_on_tx_origin_SWC_115', 'Dependence_on_predictable_environment_variable_SWC_116']


# 标记slither检测的文件并移动到对应的文件夹内
def slither_mark():
    # 源文件夹路径
    # source_folder = "../dataset/pluto_contracts_unmark"

    # 目标文件夹路径
    reentrancy_path = "../dataset/vul_kinds_marked/reentrancy"
    timestamp_path = "../dataset/vul_kinds_marked/timestamp"
    tx_origin_path = "../dataset/vul_kinds_marked/tx-origin"
    delegatecall_path = "../dataset/vul_kinds_marked/delegatecall"
    path_list = [reentrancy_path, timestamp_path, tx_origin_path, delegatecall_path]
    # 读取CSV文件并复制文件
    # CSV文件路径
    csv_file_path = "../dataset/xfuzz_slither_results.csv"
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            filename = row[0]  # 第一列是文件名

            error_finding = row[9]
            for s in vul_slither:
                if s in error_finding:
                    print("s:", s)
                    destination_file_path, score = process.extractOne(s, path_list)
                    print("destination_file_path:", destination_file_path)
                    source_file_path = "../" + filename

                    # 复制文件
                    shutil.copy(source_file_path, destination_file_path)

# 标记mythril检测的文件并移动到对应的文件夹内
def mythril_mark():
    # 源文件夹路径
    # source_folder = "../dataset/pluto_contracts_unmark"

    # 目标文件夹路径
    timestamp_path = "../dataset/vul_kinds_marked/timestamp"
    tx_origin_path = "../dataset/vul_kinds_marked/tx-origin"
    delegatecall_path = "../dataset/vul_kinds_marked/delegatecall"
    overflow_path = "../dataset/vul_kinds_marked/overflow"
    # path_list = [reentrancy_path, timestamp_path, tx_origin_path, delegatecall_path]
    # 读取CSV文件并复制文件
    # CSV文件路径
    csv_file_path = "../dataset/results-mythril.csv"  # 替换为您的CSV文件路径
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            filename = row[0]  # 第一列是文件名

            error_finding = row[9]
            for s in vul_mythril:
                if s in error_finding:
                    print("s:", s)
                    if s == "Integer_Arithmetic_Bugs_SWC_101":
                        destination_file_path = overflow_path
                    elif s == "Delegatecall_to_user_supplied_address_SWC_112":
                        destination_file_path = delegatecall_path
                    elif s == "Dependence_on_tx_origin_SWC_115":
                        destination_file_path = tx_origin_path
                    elif s == "Dependence_on_predictable_environment_variable_SWC_116":
                        destination_file_path = timestamp_path

                    print("destination_file_path:", destination_file_path)
                    source_file_path = "../" + filename

                    if os.path.exists(destination_file_path + "/" + source_file_path.split('/')[-1]):
                        print(f"文件 '{destination_file_path}' 存在。")
                    else:
                        # 复制文件
                        shutil.copy(source_file_path, destination_file_path)


def rename_files_in_folder(folder_path, new_prefix):
    # 遍历文件夹中的所有文件
    for index, filename in enumerate(os.listdir(folder_path)):
        # 构造新的文件名
        new_filename = f"{new_prefix}_{index + 1}.sol"

        # 构造文件的完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f"重命名文件 '{old_filepath}' 为 '{new_filepath}'")

# 统计数据集中有漏洞和无漏洞标签数
def check_col_num():

    count = 0
    for root, dirs, files in os.walk("../Source2Graph/graph_data/reentrancy/callee_node"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    if line_count > 3:
                        count += 1
            except Exception as e:
                print(f"无法读取文件 {file_path}：{e}")
    print("count:", count)

if __name__ == '__main__':
    # mythril_mark()
    # rename_files_in_folder("../dataset/vul_kinds_marked/tx-origin-solidify", "solidifi_buggy_txorigin")
    # check_col_num()
    slither_mark()