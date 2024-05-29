# nohup ./smartbugs -t mythril -f dataset/pluto_contracts_unmark/*.sol --processes 16 --mem-limit 12g --timeout 600 > output.txt 2>&1 &
from enum import Enum
import csv
import shutil
import os
from fuzzywuzzy import process
vul_slither = ['reentrancy-eth', 'reentrancy-no-eth', 'tx-origin', 'reentrancy-benign', 'reentrancy-events', 'timestamp',
       'reentrancy-unlimited-gas', 'controlled-delegatecall', 'delegatecall-loop']

vul_mythril = []
# class Vulnerability(Enum):
#     REENTRANCY_ETH = 1
#     REENTRANCY_NO_ETH = 1


def mark():

       # 源文件夹路径
       # source_folder = "../dataset/pluto_contracts_unmark"

       # 目标文件夹路径
       reentrancy_path = "../dataset/vul_kinds_marked/reentrancy-pluto-newmark"
       timestamp_path = "../dataset/vul_kinds_marked/timestamp-pluto-newmark"
       tx_origin__path = "../dataset/vul_kinds_marked/tx-origin-pluto-newmark"
       delegatecall_path = "../dataset/vul_kinds_marked/delegatecall-pluto-newmark"
       path_list = [reentrancy_path, timestamp_path, tx_origin__path, delegatecall_path]
       # 读取CSV文件并复制文件
       # CSV文件路径
       csv_file_path = "../dataset/slither_results.csv"  # 替换为您的CSV文件路径
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
                                   shutil.copyfile(source_file_path, destination_file_path)


if __name__ == '__main__':
    mark()


