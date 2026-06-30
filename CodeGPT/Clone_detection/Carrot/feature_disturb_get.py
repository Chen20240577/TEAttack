# -*- coding: utf-8 -*-
import ast
import glob
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
        # # 调整输入形状
        # input_ids = input_ids.view(-1, args.block_size).to(args.device)
        # attention_mask = input_ids.ne(tokenizer.pad_token_id or 1)

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


class Args:
    def __init__(self):
        self.mlm = False
        self.mlm_probability = 0.15
        self.block_size = 512
        self.device = torch.device("cuda")
        self.num_labels = 2
        self.do_lower_case = False
        self.cache_dir = None
        self.output_dir = "../saved_models"
        self.model_type = "gpt2"
        self.config_name = "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"
        self.model_name_or_path = "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"
        self.tokenizer_name = "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"
        self.seed = 123456


def main():
    # 创建模拟的args对象
    args = Args()

    set_seed(args.seed)

    # 模型初始化
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name, num_labels=args.num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 用结束符作为填充符
    config.pad_token_id = tokenizer.pad_token_id

    model = Model(model_class.from_pretrained(args.model_name_or_path, config=config),
                  config, tokenizer, args)

    # 加载模型权重
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_acc_model/model.bin')))
    model.to(args.device)
    model.eval()

    # 数据加载与验证
    df = pd.read_csv('./analysis/disturbed.csv').dropna()
    df['Original Code'] = df['Original Code'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # 清除已有的临时文件
    for f in glob.glob('./analysis/features_disturbed_rank_*.npz'):
        try:
            os.remove(f)
        except:
            pass

    # 处理每个Rank的分组
    grouped = df.groupby('Rank')
    valid_ranks = 0

    # 按组处理数据
    for rank, group in tqdm(grouped, desc="Processing Ranks"):
        # 数据一致性验证
        if len(group['Original Code'].apply(str).unique()) != 1:
            continue

        original_code = group['Original Code'].iloc[0]
        true_label = group['True label'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(original_code[2], original_code[0], true_label, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        disturb_features = []
        disturb_codes = group['Disturb Code'].tolist()

        # 处理扰动样本
        for disturb_code in disturb_codes:
            disturb_feat = extract_features(disturb_code, original_code[0], true_label, tokenizer, model, args)
            if disturb_feat is not None and disturb_feat.ndim == 1:
                disturb_features.append(disturb_feat)

        # 如果有有效特征则保存
        if disturb_features:
            n = len(disturb_features)
            # 立即保存当前Rank的特征
            np.savez_compressed(
                f'./analysis/features_disturbed_rank_{rank}.npz',
                ranks=np.array([rank] * n, dtype=np.int32),
                clean_features=np.array([clean_feat] * n, dtype=np.float32),
                disturb_features=np.array(disturb_features, dtype=np.float32)
            )
            valid_ranks += 1

    # 获取所有临时文件进行合并
    all_ranks = []
    all_clean = []
    all_disturb = []

    # 合并文件
    if valid_ranks > 0:
        files = glob.glob('./analysis/features_disturbed_rank_*.npz')
        for file_path in files:
            try:
                data = np.load(file_path)
                all_ranks.append(data['ranks'])
                all_clean.append(data['clean_features'])
                all_disturb.append(data['disturb_features'])
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {str(e)}")

        # 保存最终文件
        if all_ranks:
            all_ranks = np.concatenate(all_ranks)
            all_clean = np.concatenate(all_clean)
            all_disturb = np.concatenate(all_disturb)

            np.savez_compressed(
                './analysis/features_disturbed.npz',
                ranks=all_ranks,
                clean_features=all_clean,
                disturb_features=all_disturb
            )
            print(f"成功保存 {len(all_ranks)} 对特征")

            # 清理临时文件
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass
        else:
            print("没有有效数据保存")
    else:
        print("无有效Rank处理")


if __name__ == '__main__':
    main()
