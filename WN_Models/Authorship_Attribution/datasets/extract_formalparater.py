import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import pandas as pd
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
# 提取形式参数（函数参数）
# def extract_formal_parameters(node, code_bytes, lang):
#     parameters = []
#
#     # 通用提取规则
#     if node.type == 'identifier' and node.parent and node.parent.type == 'parameter':
#         param_name = code_bytes[node.start_byte:node.end_byte].decode('utf8')
#         parameters.append(param_name)
#
#     # 语言特定规则
#     if lang == 'java':
#         if node.type in ['formal_parameter', 'spread_parameter']:
#             for child in node.children:
#                 if child.type == 'identifier':
#                     param_name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     parameters.append(param_name)
#
#     elif lang == 'python':
#         if node.type == 'parameters':
#             for child in node.children:
#                 if child.type == 'identifier':
#                     param_name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     parameters.append(param_name)
#
#     elif lang == 'c' or lang == 'cpp':
#         if node.type == 'parameter_declaration':
#             for child in node.children:
#                 if child.type == 'identifier':
#                     param_name = code_bytes[child.start_byte:child.end_byte].decode('utf8')
#                     parameters.append(param_name)
#
#     # 递归处理子节点
#     for child in node.children:
#         parameters.extend(extract_formal_parameters(child, code_bytes, lang))
#
#     return parameters
def extract_formal_params(file, save_path, lang):
    with open(file, 'r') as f_code:
        param_list = []
        index_list = []
        count = 0

        for line in f_code:
            code = line.strip()
            if not code:
                continue

            # 使用您的标识符提取函数
            try:
                identifiers, orig_code_tokens = get_identifiers(code, lang)
                variable_names = [iden[0] for iden in identifiers]

                # 提取形式参数 - 这里需要根据您的需求调整逻辑
                formal_params = []
                if '(' in code and ')' in code:
                    # 根据语言类型提取参数
                    if lang == 'java':
                        # Java参数提取逻辑
                        if '(' in code and ')' in code:
                            params_str = code.split('(', 1)[1].split(')', 1)[0]
                            # 从提取的变量名中筛选可能是参数的
                            for var in variable_names:
                                if var in params_str:
                                    formal_params.append(var)

                    elif lang == 'python' and 'def ' in code:
                        # Python参数提取逻辑
                        def_index = code.find('def ')
                        if def_index != -1:
                            func_part = code[def_index + 4:].strip()
                            if '(' in func_part and ')' in func_part:
                                params_str = func_part.split('(', 1)[1].split(')', 1)[0]
                                for var in variable_names:
                                    if var in params_str:
                                        formal_params.append(var)

                    elif lang in ['c', 'cpp']:
                        # C语言参数提取逻辑
                        if '(' in code and ')' in code:
                            params_str = code.split('(', 1)[1].split(')', 1)[0]
                            for var in variable_names:
                                if var in params_str:
                                    formal_params.append(var)

            except Exception as e:
                print(f"提取标识符错误 ({lang}): {e}")
                formal_params = []

            param_list.append(formal_params)
            index_list.append(count)
            count += 1
            print(f'Processed {count}')

    data_var = pd.DataFrame({'id': index_list, 'formal_parameters': param_list})
    data_var.to_pickle(save_path)


if __name__ == '__main__':
    root = '../../../Datasets/Authorship_Attribution/codebert-mlm'

    # 多语言支持配置
    config = {
        # "java": {
        #     "file": root + '/processed_gcjpy/valid_java.txt',
        #     "save_path": './datasets/java_var_all.pkl',
        # },
        "python": {
            "file": root + '/processed_gcjpy/valid.txt',
            "save_path": 'formalParameter_for_everyCode.pkl',
        },
        # "c": {
        #     "file": root + '/processed_gcjpy/valid_c.txt',
        #     "save_path": './datasets/c_var_all.pkl',
        # }
    }

    for lang, paths in config.items():
        print('extract formal parameters :')
        print(f'Processing {lang} files...')
        extract_formal_params(
            file=paths["file"],
            save_path=paths["save_path"],
            lang=lang
        )
    print('extract end!')
