# -*- coding: utf-8 -*-
import argparse
import ast
import os
# from csv_attacker import conduct_example
# 克隆检测任务特殊，重写 conduct_example
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

retval = os.getcwd()

import random
import logging
import warnings
import torch
import numpy as np

from model import Model
from run_parser import get_identifiers

from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
}

logger = logging.getLogger(__name__)

# 模型名称到文件缩写的映射
MODEL_ABBREVIATIONS = {
    "CodeBert": "Bert",
    "GCBert": "GCBert",  # 如果没有缩写，使用全名
    "CodeT5": "T5",  # 假设RoBERTa也使用Bert作为缩写
    "CodeGPT": "GPT",
    # 可以继续添加其他模型的映射
}


def get_model_abbreviation(model_name):
    """根据模型名称获取文件中的缩写"""
    return MODEL_ABBREVIATIONS.get(model_name, model_name)  # 如果没有映射，使用原模型名称


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 label,
                 url
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.url = url


def convert_examples_to_features(code_tokens, label, url, tokenizer, args):
    # source
    code_tokens = code_tokens[:args.block_size - 2]
    code_tokens = [tokenizer.eos_token] + code_tokens + [tokenizer.eos_token]

    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    attention_mask = (code_ids != 0)

    return InputFeatures(code_tokens, code_ids, attention_mask, label, url)


def conduct_example(code, url, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(code)
    example = convert_examples_to_features(code_tokens, label, url, tokenizer, args)

    return torch.tensor(example.input_ids), torch.tensor(example.label)


def extract_features(code, url, label, tokenizer, model, args):
    """通用特征提取函数"""
    try:
        code = str(code)
        url = str(url)
        # 生成模型输入
        example = conduct_example(code, url, int(label), tokenizer, args)
        input_ids, _ = example

        input_ids = input_ids.view(-1, args.block_size).to(args.device)
        # 如果切分后有多块，只取第一块
        if input_ids.size(0) > 1:
            input_ids = input_ids[0:1, :]  # 保留batch维度
        attention_mask = (input_ids != 0)

        with torch.no_grad():
            # GPT2ForSequenceClassification -> encoder.transformer 是 GPT2Model
            outputs = model.encoder.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

            # 取最后一个 token 的 hidden state
            code_vector = hidden_states[:, -1, :]  # [batch, hidden_dim]

            return code_vector.cpu().numpy().squeeze()

    except Exception as e:
        print(f"特征提取失败 (代码: {type(code)}|{len(code) if isinstance(code, str) else 'N/A'}")
        return None


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()

    ## Required parameters - 必需的路径参数
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="任务名称，如: Authorship_Attribution")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="模型名称，如: CodeBert")
    parser.add_argument("--victim", default=None, type=str, required=True,
                        help="模型名称，如: CodeBert")
    parser.add_argument("--method", default=None, type=str, required=True,
                        help="方法名称，如: ACCENT")

    ## Optional parameters - 可选参数
    parser.add_argument("--base_path", default="../../../TransfAEs", type=str,
                        help="基础输入路径")
    parser.add_argument("--output_base", default="./transfer", type=str,
                        help="基础输出路径")
    parser.add_argument("--output_dir", default="../saved_models", type=str,
                        help="模型保存目录")

    # 模型相关参数
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="模型类型")
    parser.add_argument("--config_name", default="../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2",
                        type=str,
                        help="配置文件路径")
    parser.add_argument("--model_name_or_path",
                        default="../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2", type=str,
                        help="模型路径")
    parser.add_argument("--tokenizer_name", default="../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2",
                        type=str,
                        help="tokenizer路径")

    # 训练/评估参数
    parser.add_argument("--block_size", default=512, type=int,
                        help="输入序列最大长度")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="标签数量")
    parser.add_argument("--seed", default=123456, type=int,
                        help="随机种子")

    args = parser.parse_args()

    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取模型在文件名中的缩写
    model_abbr = get_model_abbreviation(args.victim)
    # 构建完整的输入输出路径
    input_file = os.path.join(args.base_path, args.task, args.model, f'{args.method}_{model_abbr}_transfer.csv')
    output_dir = os.path.join(args.output_base, args.victim)

    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)

    # 模型初始化
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name, num_labels=args.num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    config.pad_token_id = tokenizer.eos_token_id

    model = Model(model_class.from_pretrained(args.model_name_or_path, config=config),
                  config, tokenizer, args)

    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_acc_model/model.bin')))
    model.to(args.device)
    model.eval()

    # 数据加载与验证
    try:
        df = pd.read_csv(input_file).dropna()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return

    # 验证必要的列是否存在
    required_columns = ['Rank', 'Original Code', 'Disturb Code', 'True label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"数据集缺少必要的列，现有列: {df.columns.tolist()}")

    df['Original Code'] = df['Original Code'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    df = df.astype({
        'Rank': int,
        'Original Code': object,  # 使用object类型存储列表
        'Disturb Code': str,
        'True label': bool
    })

    # 存储结构初始化
    ranks = []
    clean_feats = []
    attack_feats = []
    success_flags = []

    # 按Rank分组处理
    for rank, group in tqdm(df.groupby('Rank'), desc="Processing Groups"):
        # 数据一致性验证 - 使用字符串表示进行比较
        original_codes = group['Original Code'].apply(str).unique()
        true_labels = group['True label'].unique()

        if len(original_codes) != 1 or len(true_labels) != 1:
            print(f"跳过不一致的Index {rank}")
            continue

        original_code = group['Original Code'].iloc[0]
        success_flag = group['True label'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(original_code[2], original_code[0], 1, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        # 处理Disturb Code样本
        dist_code = group['Disturb Code'].iloc[0]
        dist_feat = extract_features(dist_code, original_code[0], 1, tokenizer, model, args)
        if dist_feat is None or dist_feat.ndim != 1:
            continue

        # 存储特征对和迁移结果标志
        ranks.append(rank)
        clean_feats.append(clean_feat)
        attack_feats.append(dist_feat)
        success_flags.append(success_flag)

    # 保存为单个压缩文件
    if len(ranks) > 0:
        # 统计成功和失败的样本数量
        success_count = sum(success_flags)
        failure_count = len(success_flags) - success_count

        output_file = os.path.join(output_dir, 'transfer_features.npz')
        np.savez_compressed(
            output_file,
            ranks=np.array(ranks, dtype=np.int32),
            clean_features=np.array(clean_feats, dtype=np.float32),
            attack_features=np.array(attack_feats, dtype=np.float32),
            success_flags=np.array(success_flags, dtype=bool)
        )
        print(f"成功保存{len(ranks)}对特征到单个文件（成功迁移: {success_count}, 失败迁移: {failure_count}）")
        print(f"特征维度：{clean_feats[0].shape}")
        print(f"输出文件：'{output_file}'")
    else:
        print("无有效数据保存")


if __name__ == '__main__':
    main()
