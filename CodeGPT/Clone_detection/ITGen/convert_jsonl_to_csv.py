# -*- coding: utf-8 -*-
import json
import csv
import re
import os
import sys
import torch
import argparse
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

# 添加必要的路径
sys.path.append('../../../')
sys.path.append('../saved_models/')
sys.path.append('../../../python_parser')
from model import Model
from run import TextDataset

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer)
}

import pickle

def jsonl_to_csv_simple_with_extras():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    jsonl_path = "./analysis/attack_itgen_all.jsonl"
    csv_path = "../../../AdvExamples/Clone_detection/ITGen_GPT.csv"
    args.eval_data_file = "../../../Datasets/Clone_detection/codegpt-small/data_folder/test_sampled_0_500.txt"
    args.model_path = "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"
    args.tokenizer_name = "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"
    args.model_type = "gpt2"
    args.num_labels = 2
    args.label_idx = 1
    args.block_size = 512
    args.code_length = 384
    args.data_flow_length = 128
    args.eval_batch_size = 8
    args.output_dir = "../saved_models"  # 请根据实际情况修改
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"错误: 找不到JSONL文件: {jsonl_path}")
        return False

    # 运行增强版转换
    return enhanced_jsonl_to_csv(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        subs_path=args.eval_data_file,
        args=args
    )


def get_code_pairs(file_path):
    postfix = file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def load_tokenizer(args):
    """加载预训练模型和tokenizer"""
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    return tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=False)


def enhanced_jsonl_to_csv(jsonl_path, csv_path, subs_path, args):
    # 确保当前工作目录正确
    print(f"当前工作目录: {os.getcwd()}")
    print(f"输入文件: {jsonl_path}")
    print(f"输出文件: {csv_path}")

    # 检查输入文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"错误: 找不到JSONL文件: {jsonl_path}")
        return False

    tokenizer = load_tokenizer(args)

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # 加载原始代码
    source_codes = get_code_pairs(subs_path)
    
    assert len(source_codes) == len(eval_dataset)
    
    if source_codes is None:
        print("警告: 无法加载原始代码，相关字段将留空")

    # 创建输出目录
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
        print(f"已创建目录: {csv_dir}")

    # 读取JSONL文件
    data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"读取JSONL文件失败: {e}")
        return False

    if not data:
        print("JSONL文件为空或格式错误")
        return False

    # CSV表头
    headers = [
        "Index", "Original Code", "Program Length", "Adversarial Code",
        "True Label", "Original Prediction", "Adv Prediction", "Is Success",
        "Extracted Names", "Importance Score", "No. Changed Names",
        "No. Changed Tokens", "Replaced Names", "Attack Type",
        "Query Times", "Time Cost"
    ]

    # 写入CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            success_count = 0
            fail_count = 0

            for i, item in enumerate(data):
                index = item.get("Index", i)
                if i %50==0:
                    print("\n i :",i)
                # ① 获取Original Code
                # 先从JSONL中获取，如果为null或空，则从source_codes中获取
                orig_code_from_jsonl = item.get("Original Code", "")
                if (not orig_code_from_jsonl or orig_code_from_jsonl == "null" or
                    orig_code_from_jsonl is None) and source_codes:
                    # 使用Index作为索引（注意：Index可能从1开始，而列表索引从0开始）
                    idx = int(index) - 1 if isinstance(index, (int, str)) and str(index).isdigit() else i
                    if 0 <= idx < len(source_codes):
                        orig_code = source_codes[idx]
                    else:
                        orig_code = orig_code_from_jsonl
                else:
                    orig_code = orig_code_from_jsonl

                prog_len = item.get("Program Length", 0)
                adv_code = item.get("Adversarial Code", "")
                replace_info = item.get("Replaced Identifiers", "")
                query_times = item.get("Query Times", 0)
                time_cost = item.get("Time Cost", 0.0)
                attack_type = item.get("Type", "")

                # 判断攻击是否成功
                is_success = 1 if adv_code != "" else 0

                if is_success:
                    success_count += 1
                else:
                    fail_count += 1

                # 提取原始变量名
                extracted_names = ""
                if replace_info and replace_info != "null":
                    matches = re.findall(r'([^:,]+):[^,]*', replace_info)
                    extracted_names = ",".join(matches) if matches else ""

                # 计算改变的变量名数量
                if replace_info and replace_info != "null":
                    pairs = [p for p in replace_info.split(',') if p.strip()]
                    changed_names = len(pairs)
                else:
                    changed_names = 0

                # 计算改变的token数量
                changed_tokens = 0
                if orig_code and orig_code != "null":
                    if adv_code and adv_code != "null":  # 有对抗样本的情况
                        orig_lines = str(orig_code).strip().split('\n')
                        adv_lines = str(adv_code).strip().split('\n')

                        # 比较对应行的差异
                        for j in range(min(len(orig_lines), len(adv_lines))):
                            if orig_lines[j] != adv_lines[j]:
                                orig_tokens = orig_lines[j].split()
                                adv_tokens = adv_lines[j].split()
                                changed_tokens += sum(1 for k in range(min(len(orig_tokens), len(adv_tokens)))
                                                      if orig_tokens[k] != adv_tokens[k])
                                changed_tokens += abs(len(orig_tokens) - len(adv_tokens))

                        # 加上长度差异的行
                        for j in range(min(len(orig_lines), len(adv_lines)), max(len(orig_lines), len(adv_lines))):
                            if j < len(orig_lines):
                                changed_tokens += len(orig_lines[j].split())
                            else:
                                changed_tokens += len(adv_lines[j].split())
                    else:  # 没有对抗样本的情况
                        changed_tokens = 0
                else:
                    changed_tokens = 0

                # 如果攻击失败，对抗代码为空
                if attack_type == "0" or attack_type == "":
                    adv_code = ""
                    attack_type_display = ""
                else:
                    attack_type_display = attack_type

                # ② 获取True Label
                example =eval_dataset[index]

                true_label = example[args.label_idx].item()

                # 写入一行
                csv_row = [
                    index,
                    orig_code,
                    prog_len,
                    adv_code,
                    true_label,  # True Label
                    "",  # Original Prediction (暂时留空)
                    "",  # Adv Prediction (暂时留空)
                    is_success,
                    extracted_names,
                    "",  # Importance Score
                    changed_names,
                    changed_tokens,
                    replace_info if replace_info and replace_info != "null" else "",
                    attack_type_display,
                    query_times,
                    time_cost
                ]
                writer.writerow(csv_row)

        print(f"\n转换完成！")
        print(f"总记录数: {len(data)}")
        if len(data) > 0:
            print(f"成功率: {success_count / len(data) * 100:.2f}%")
        print(f"输入: {jsonl_path}")
        print(f"输出: {csv_path}")
        return True

    except Exception as e:
        print(f"写入CSV文件失败: {e}")
        return False


if __name__ == "__main__":
    # 使用方法1: 直接运行简单版本
    jsonl_to_csv_simple_with_extras()

