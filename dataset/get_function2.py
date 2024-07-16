import os
import re


def extract_functions(content):
    function_pattern = re.compile(
        r'function\s+[\w]+\s*\(.*?\)\s*(public|private|internal|external)?\s*(returns\s*\(.*?\))?\s*\{')
    end_brace_pattern = re.compile(r'\}')

    functions = set()
    inside_function = False
    function_content = []

    for line in content.splitlines():
        if not inside_function:
            match = function_pattern.search(line)
            if match:
                inside_function = True
                function_content.append(line)
        else:
            function_content.append(line)
            if end_brace_pattern.search(line):
                function_def = "\n".join(function_content)
                functions.add(function_def)
                inside_function = False
                function_content = []

    return functions


def get_function_definitions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return extract_functions(content)


def count_unique_functions_in_folder(folder_path):
    unique_functions = set()
    count = 1
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.sol'):
                file_path = os.path.join(root, file)
                unique_functions.update(get_function_definitions(file_path))
                count += 1
                print("count:", count)
    return len(unique_functions)


# Example usage
folder_path = './xfuzz_contracts_unmark'
unique_function_count = count_unique_functions_in_folder(folder_path)
print(f'Total number of unique functions in all .sol files: {unique_function_count}')
