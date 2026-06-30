import sys

from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

import csv
import torch
from model import Model

from run_parser import get_identifiers, get_example


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(index, code, label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, index, int(label))


def conduct_example(index, code, label, tokenizer, args):
    example = convert_examples_to_features(index, code, label, tokenizer, args)
    return torch.tensor(example.input_ids), torch.tensor(example.label)


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
        # 增加字段大小限制
        csv.field_size_limit(sys.maxsize)

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
            index = row.get('Index', '').strip()
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

            orig_example = conduct_example(index, orig_code, int(true_label), tokenizer, args)
            orig_logits, orig_preds = self.model_tgt.get_results([orig_example], self.args.eval_batch_size)
            pre_label = orig_preds[0]

            if pre_label != true_label:
                original_fail_cnt += 1
                continue

            # 构造示例（假设示例格式与之前相同）
            adv_example = conduct_example(index, adv_code, int(true_label), tokenizer, args)
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
