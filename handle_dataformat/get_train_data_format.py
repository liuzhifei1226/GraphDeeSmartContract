import Source2Graph.Graph.ExtractGraphCallee
import os


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
        file_path = directory_path + '/' + f
        if not os.path.isfile("../Source2Graph/graph_data/reentrancy/callee_node"+ f):
            Source2Graph.Graph.ExtractGraphCallee.getGraph(file_path)
        else:
            continue

def source2vec():


# 示例用法
if __name__ == '__main__':
    directory_path = "../dataset/vul_kinds_marked/reentrancy-pluto-newmark"
    files = list_files_in_directory(directory_path)
    # print("文件列表:", files)
    # source2graph(directory_path, files)
