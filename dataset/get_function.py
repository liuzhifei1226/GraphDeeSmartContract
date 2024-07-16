import os
import re


def get_function_definitions(file_path):
    # This regex is designed to capture the entire function definition, including the body.
    function_pattern = re.compile(
        r'function\s+[\w]+\s*\(.*?\)\s*(public|private|internal|external)?\s*(returns\s*\(.*?\))?\s*\{(?:[^{}]*|\{(?:[^{}]*|\{[^{}]*\})*\})*\}')
    functions = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = function_pattern.findall(content)
        for match in matches:
            functions.add(match[0])

    return functions


def count_unique_functions_in_folder(folder_path):
    unique_functions = set()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.sol'):
                file_path = os.path.join(root, file)
                unique_functions.update(get_function_definitions(file_path))

    return len(unique_functions)


# Example usage
folder_path = './pluto_contracts_unmark'
unique_function_count = count_unique_functions_in_folder(folder_path)
print(f'Total number of unique functions in all .sol files: {unique_function_count}')
