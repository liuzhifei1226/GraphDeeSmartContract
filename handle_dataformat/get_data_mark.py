# nohup ./smartbugs -t mythril -f dataset/pluto_contracts_unmark/*.sol --processes 16 --mem-limit 12g --timeout 600 > output.txt 2>&1 &
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


# class Vulnerability(Enum):
#     REENTRANCY_ETH = 1
#     REENTRANCY_NO_ETH = 1


def slither_mark():
    # 源文件夹路径
    # source_folder = "../dataset/pluto_contracts_unmark"

    # 目标文件夹路径
    reentrancy_path = "../dataset/vul_kinds_marked/reentrancy-pluto-newmark"
    timestamp_path = "../dataset/vul_kinds_marked/timestamp-pluto-newmark"
    tx_origin_path = "../dataset/vul_kinds_marked/tx-origin-pluto-newmark"
    delegatecall_path = "../dataset/vul_kinds_marked/delegatecall-pluto-newmark"
    path_list = [reentrancy_path, timestamp_path, tx_origin_path, delegatecall_path]
    # 读取CSV文件并复制文件
    # CSV文件路径
    csv_file_path = "../dataset/slither_results_7799.csv"  # 替换为您的CSV文件路径
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


def mythril_mark():
    # 源文件夹路径
    # source_folder = "../dataset/pluto_contracts_unmark"

    # 目标文件夹路径
    timestamp_path = "../dataset/vul_kinds_marked/timestamp-pluto-newmark"
    tx_origin_path = "../dataset/vul_kinds_marked/tx-origin-pluto-newmark"
    delegatecall_path = "../dataset/vul_kinds_marked/delegatecall-pluto-newmark"
    overflow_path = "../dataset/vul_kinds_marked/overflow-pluto-newmark"
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


if __name__ == '__main__':
    # mythril_mark()
    rename_files_in_folder("../dataset/vul_kinds_marked/tx-origin-solidify", "solidifi_buggy_txorigin")
