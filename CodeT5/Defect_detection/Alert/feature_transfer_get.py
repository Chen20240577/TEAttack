# -*- coding: utf-8 -*-
import argparse
import os
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

from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
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


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_features(rank, code, label, tokenizer, model, args):
    try:
        # 生成模型输入
        example = conduct_example(rank, code, int(label), tokenizer, args)
        input_ids, _ = example

        # 调整输入形状
        input_ids = input_ids.view(-1, args.block_size).to(args.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # 使用模型的基础编码器获取隐藏状态
        with torch.no_grad():
            # 关键修改：直接访问基础模型（roberta）而非分类头
            outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # 为解码器提供输入
                output_hidden_states=True,
                return_dict=True
            )
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
            eos_mask = input_ids.eq(model.config.eos_token_id)

            # 处理无eos token的情况
            if eos_mask.sum().item() == 0:
                # 使用序列末尾作为替代
                eos_mask[:, -1] = True

            # 提取特征 (确保正确索引)
            # 原始方法: features = decoder_hidden_states[eos_mask, :]
            # 改为更安全的方法:
            batch_size = decoder_hidden_states.size(0)
            features_list = []
            for i in range(batch_size):
                # 获取当前样本的eos位置
                sample_eos_mask = eos_mask[i]
                # 获取eos位置的隐藏状态
                sample_features = decoder_hidden_states[i, sample_eos_mask]
                # 取最后一个eos位置的特征
                if sample_features.dim() > 1:
                    features_list.append(sample_features[-1])
                else:
                    features_list.append(sample_features)

            cls_feature = torch.stack(features_list).squeeze(0).cpu().numpy()
            return cls_feature

    except Exception as e:
        print(f"特征提取失败: {str(e)}")
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
    # 模型相关参数
    parser.add_argument("--model_type", default="t5", type=str,
                        help="模型类型")
    parser.add_argument("--config_name", default="../../../../../Models/Salesforce/codet5-base", type=str,
                        help="配置文件路径")
    parser.add_argument("--model_name_or_path", default="../../../../../Models/Salesforce/codet5-base", type=str,
                        help="模型路径")
    parser.add_argument("--tokenizer_name", default="../../../../../Models/Salesforce/codet5-base", type=str,
                        help="tokenizer路径")

    # 训练/评估参数
    parser.add_argument("--block_size", default=512, type=int,
                        help="输入序列最大长度")
    parser.add_argument("--num_labels", default=1, type=int,
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
        print(f"数据集缺少必要的列，现有列: {df.columns.tolist()}")
        return

    # 数据类型转换
    df = df.astype({'Rank': int, 'Original Code': str, 'Disturb Code': str, 'True label': bool})

    # 存储结构初始化
    ranks = []
    clean_feats = []
    attack_feats = []
    success_flags = []

    # 按Rank分组处理
    for rank, group in tqdm(df.groupby('Rank'), desc="Processing Groups"):
        # 数据一致性验证
        if len(group['Original Code'].unique()) != 1 or len(group['True label'].unique()) != 1:
            print(f"跳过不一致的Rank {rank}")
            continue

        original_code = group['Original Code'].iloc[0]
        success_flag = group['True label'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(rank, original_code, 1, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        # 处理Disturb Code样本
        dist_code = group['Disturb Code'].iloc[0]
        dist_feat = extract_features(rank, dist_code, 1, tokenizer, model, args)
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
