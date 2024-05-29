import os

# 定义文件夹路径
folder_path = '../dataset/pluto_contracts_unmark'

files_to_delete = []

# 获取文件夹下的所有文件列表
file_list = os.listdir(folder_path)

# 遍历每个文件
for file_name in file_list:
    # 拼接文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 判断文件是否为普通文件并且可读
    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
        # 打开文件进行读取，使用utf-8编码
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # 读取文件内容
            content = file.read()

            # 判断文件内容是否包含字符串"pragma"
            if "pragma" not in content:
                # 将需要删除的文件路径记录到列表中
                files_to_delete.append(file_path)
                print(f"Added {file_name} to the list for deletion as it doesn't contain 'pragma'.")
            else:
                print(f"Kept {file_name} as it contains 'pragma'.")
    else:
        print(f"Skipping {file_name} as it's not a readable file.")

# 统一删除需要删除的文件
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}.")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
