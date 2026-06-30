import json
import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import pandas as pd
import os
from run_parser import get_identifiers


# # 初始化 Tree-sitter 多语言支持
# def init_tree_sitter():
#     # 加载预编译的多语言库
#     LANGUAGE_LIB = '../../../python_parser/parser_folder/my-languages.so'
#     languages = {
#         'java': Language(LANGUAGE_LIB, 'java'),
#         'python': Language(LANGUAGE_LIB, 'python'),
#         'c': Language(LANGUAGE_LIB, 'c'),
#     }
#     return languages
#
#
# # 递归遍历 AST 并提取变量/函数名
# def extract_names(node, code_bytes, lang):
#     names = set()
#
#     # 通用提取规则
#     if node.type == 'identifier':
#         name = code_bytes[node.start_byte:node.end_byte].decode('utf8')
#         names.add(name)
#
#     # 语言特定规则
#     if lang == 'java':
#         if node.type in ['method_declaration', 'variable_declarator']:
#             for child in node.children:
#                 if child.type == 'identifier':
#                     name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     names.add(name)
#
#     elif lang == 'python':
#         if node.type in ['function_definition', 'assignment']:
#             for child in node.children:
#                 if child.type == 'identifier':
#                     name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     names.add(name)
#
#     elif lang == 'c' or lang == 'cpp':
#         if node.type in ['function_definition', 'declaration']:
#             for child in node.children:
#                 if child.type == 'identifier':
#                     name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     names.add(name)
#
#     # 递归处理子节点
#     for child in node.children:
#         names |= extract_names(child, code_bytes, lang)
#
#     return names
def extract_var_name(file, save_var_path, save_all_path, lang):
    with open(file, 'r') as f_code:
        set_var = set()
        var_list = []
        index_list = []
        count = 0

        for line in f_code:
            line = line.strip()
            if not line:
                continue

            js = json.loads(line)
            code = js['code1']  # 获取代码内容
            if not code:
                continue

            try:
                # 使用您的标识符提取函数替代Tree-sitter
                identifiers, orig_code_tokens = get_identifiers(code, lang)
                variable_names = [iden[0] for iden in identifiers]

                # 添加函数名（如果有）
                if lang == 'java' and '(' in code:
                    method_name = code.split('(')[0].split()[-1]
                    if method_name and method_name not in variable_names:
                        variable_names.append(method_name)
                elif lang == 'python' and 'def ' in code:
                    func_name = code.split('def ')[1].split('(')[0].strip()
                    if func_name and func_name not in variable_names:
                        variable_names.append(func_name)
                elif lang in ['c', 'cpp'] and '(' in code and not code.strip().startswith('#'):
                    func_name = code.split('(')[0].split()[-1]
                    if func_name and func_name not in variable_names:
                        variable_names.append(func_name)

            except Exception as e:
                print(f"提取标识符错误 ({lang}): {e}")
                variable_names = []

            var_list.append(variable_names)
            set_var.update(variable_names)
            index_list.append(count)
            count += 1
            print(f'Processed {count}')

    os.makedirs('var_name', exist_ok=True)

    data_var = pd.DataFrame({'id': index_list, 'variable': var_list})
    data_var.to_pickle(save_var_path)

    var_all_list = list(set_var)
    data_var_all = pd.DataFrame({'id': 0, 'all vars': [var_all_list]})
    data_var_all.to_pickle(save_all_path)


if __name__ == '__main__':

    root = '../../../Datasets/Clone_detection/codebert-mlm'

    # 多语言支持配置
    config = {
        "java": {
            "file": root + '/data_folder/test_subs_0_500.jsonl',
            "save_all": 'var_for_allCode.pkl',
            "save_var": 'var_for_everyCode.pkl'
        }
    }

    for lang, paths in config.items():
        print(f'Processing {lang} files...')
        extract_var_name(
            file=paths["file"],
            save_var_path=paths["save_var"],
            save_all_path=paths["save_all"],
            lang=lang
        )
