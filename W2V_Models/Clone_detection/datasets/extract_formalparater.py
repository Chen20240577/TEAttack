import json
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
#
#
# # 提取形式参数（函数参数）
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
            line = line.strip()
            if not line:
                continue

            js = json.loads(line)
            code = js['code1']  # 获取代码内容
            if not code:
                continue

            formal_params = []

            try:
                # 使用您的标识符提取函数替代Tree-sitter
                identifiers, orig_code_tokens = get_identifiers(code, lang)
                variable_names = [iden[0] for iden in identifiers]

                # 根据语言类型提取形式参数
                if '(' in code and ')' in code:
                    # 提取括号内的参数部分
                    params_str = code.split('(', 1)[1].split(')', 1)[0]

                    if lang == 'java':
                        # Java参数提取逻辑
                        params = []
                        for p in params_str.split(','):
                            p = p.strip()
                            if p:
                                # 处理带类型注解的参数
                                if 'final' in p:
                                    p = p.replace('final', '').strip()
                                parts = p.split()
                                if parts:
                                    # 取最后一个部分作为参数名
                                    param_name = parts[-1]
                                    # 去除可能的数组符号
                                    param_name = param_name.replace('[]', '')
                                    # 检查这个参数名是否在提取的标识符中
                                    if param_name in variable_names:
                                        params.append(param_name)
                        formal_params.extend(params)

                    elif lang == 'c' and '(' in code:
                        # C语言参数提取逻辑
                        params = []
                        for p in params_str.split(','):
                            p = p.strip()
                            if p:
                                # 处理指针类型参数
                                if '*' in p:
                                    p = p.replace('*', '').strip()
                                parts = p.split()
                                if parts:
                                    # 取最后一个部分作为参数名
                                    param_name = parts[-1]
                                    # 检查这个参数名是否在提取的标识符中
                                    if param_name in variable_names:
                                        params.append(param_name)
                        formal_params.extend(params)

                    elif lang == 'python' and 'def ' in code:
                        # Python参数提取逻辑
                        params = []
                        for p in params_str.split(','):
                            p = p.strip()
                            if p:
                                # 如果有默认值，只取等号前面的部分
                                if '=' in p:
                                    p = p.split('=')[0].strip()
                                # 如果有类型注解，取冒号前面的部分
                                if ':' in p:
                                    p = p.split(':')[0].strip()
                                # 参数名应该是单个标识符
                                if ' ' in p:
                                    p = p.split()[-1]
                                # 检查这个参数名是否在提取的标识符中
                                if p in variable_names:
                                    params.append(p)
                        formal_params.extend(params)

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
    root = '../../../Datasets/Clone_detection/codebert-mlm'

    # 多语言支持配置
    config = {
        "java": {
            "file": root + '/data_folder/test_subs_0_500.jsonl',
            "save_path": 'formalParameter_for_everyCode.pkl',
        },
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
