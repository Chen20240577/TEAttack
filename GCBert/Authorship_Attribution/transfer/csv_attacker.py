import csv
import sys

from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

import torch
import numpy as np

from run_parser import get_identifiers, get_example
# from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
# from parser import (remove_comments_and_docstrings,
#                    tree_to_token_index,
#                    index_to_code_token,
#                    tree_to_variable_index)
from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG import DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)

from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    # LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_dataflow(code, parser, lang):
    # remove comments
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 label

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.label = label


def convert_examples_to_features(code, label, tokenizer, args):
    # source
    # code=' '.join(js['func'].split())
    parser = parsers["python"]
    code_tokens, dfg = extract_dataflow(code, parser, "python")

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)


def conduct_example(code, label, tokenizer, args):
    # 构造示例（假设示例格式与之前相同）
    example = convert_examples_to_features(code, label, tokenizer, args)

    # 转换为tensor格式
    attn_mask = np.zeros((args.code_length + args.data_flow_length,
                          args.code_length + args.data_flow_length), dtype=bool)
    # calculate begin index of node and max length of input

    node_index = sum([i > 1 for i in example.position_idx])
    max_length = sum([i != 1 for i in example.position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(example.input_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(example.dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(example.dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(example.position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (torch.tensor(example.input_ids),
            torch.tensor(attn_mask),
            torch.tensor(example.position_idx),
            torch.tensor(example.label))


class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def csv_attack(self, csv_path, args, tokenizer, transfer_recoder):
        """
                从 CSV 文件中提取已经成功的对抗示例，验证模型是否会被这些对抗示例攻击，并计算对抗示例的成功率。

                参数:
                    csv_path (str): 包含对抗示例的 CSV 文件路径。
                """
        # 打开 CSV 文件
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)  # 读取所有行

        success_attack = 0  # 成功攻击次数
        total_cnt = 0  # 总对抗示例数量
        invalid_cnt = 0  # 无效对抗示例数量
        original_fail_cnt = 0  # 原始预测错误的样本数量
        # 遍历每一行
        for row in tqdm(rows, desc="验证对抗示例"):
            # 读取对抗代码和其他信息
            index = row.get('Index')
            adv_code = row.get('Adversarial Code', '').strip()  # 获取对抗代码，并去除空白字符
            true_label = row.get('True Label', '').strip()  # 真实标签
            orig_code = row.get('Original Code', '').strip()
            orig_label = row.get('Original Prediction', '').strip()  # 原始预测标签

            # 检查数据是否有效
            if not adv_code or not true_label or not orig_code or not orig_label:
                invalid_cnt += 1
                continue  # 跳过无效数据

            try:
                true_label = int(true_label)
                orig_label = int(orig_label)
            except ValueError:
                invalid_cnt += 1
                continue  # 跳过无法转换为整数的标签

            total_cnt += 1

            # 构造示例（假设示例格式与之前相同）
            orig_example = conduct_example(orig_code, int(true_label), tokenizer, args)
            orig_logits, orig_preds = self.model_tgt.get_results([orig_example], self.args.eval_batch_size)
            pre_label = orig_preds[0]
            if pre_label != true_label:
                original_fail_cnt += 1
                continue

            # 构造示例（假设示例格式与之前相同）
            adv_example = conduct_example(adv_code, int(true_label), tokenizer, args)
            # 使用目标模型对对抗示例进行预测
            logits, preds = self.model_tgt.get_results([adv_example], self.args.eval_batch_size)
            temp_label = preds[0]

            # 统计攻击结果
            # 迁移成功的记录True，失败的记录False
            if temp_label != true_label:
                transfer_recoder.write(index, orig_code, adv_code, True)
                success_attack += 1
            else:
                transfer_recoder.write(index, orig_code, adv_code, False)

        # 提取 CSV 文件名和基础模型名称
        csv_name = csv_path.split('/')[-1].split('.')[0]  # 提取文件名，去掉路径和扩展名
        base_model_name = args.model_name_or_path.split('/')[-1]  # 提取基础模型名称
        # 输出统计信息
        print(f"对抗示例验证完成！")
        print(f"对抗示例来源: {csv_name}")
        print(f"攻击的模型: {base_model_name}")
        print(f"对抗示例总量: {total_cnt}")
        print(f"原始预测错误的样本数量: {original_fail_cnt}")
        print(f"有效对抗示例总量: {total_cnt - original_fail_cnt}")
        print(f"成功攻击次数: {success_attack}")
        if total_cnt > 0:
            denominator = total_cnt - original_fail_cnt
            success_rate = success_attack / denominator if denominator != 0 else 0.0
            print(f"对抗示例攻击成功率: {success_rate:.4f}")
        else:
            print("未找到有效的对抗示例。")
        print(f"无效样本数量: {invalid_cnt}")
