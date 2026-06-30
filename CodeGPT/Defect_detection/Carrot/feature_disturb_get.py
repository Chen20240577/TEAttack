# -*- coding: utf-8 -*-
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

from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
}

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(index, code, label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.eos_token] + code_tokens + [tokenizer.eos_token]

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    attention_mask = (source_ids != 0)

    return InputFeatures(source_tokens, source_ids, attention_mask, index, int(label))


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
        if input_ids.size(0) > 1:
            input_ids = input_ids[0:1, :]  # 保留batch维度
        attention_mask = (input_ids != 0)

        # 使用模型的基础编码器获取隐藏状态
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
        print(f"特征提取失败: {str(e)}")
        return None


class Args:
    def __init__(self):
        self.mlm = False
        self.mlm_probability = 0.15
        self.block_size = 512
        self.device = torch.device("cuda")
        self.num_labels = 1
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
    df = df.astype({'Rank': int, 'Original Code': str, 'Disturb Code': str, 'True label': int})

    # 存储结构初始化
    ranks = []
    clean_feats = []
    disturb_feats = []

    # 按Rank分组处理（添加主进度条）
    groups = list(df.groupby('Rank'))  # 转换为列表以获取总数
    for (rank, group) in df.groupby('Rank'):
        # 数据一致性验证
        if len(group['Original Code'].unique()) != 1 or len(group['True label'].unique()) != 1:
            print(f"跳过不一致的Rank {rank}")
            continue

        original_code = group['Original Code'].iloc[0]
        true_label = group['True label'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(rank, original_code, true_label, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        # 批量处理扰动样本（添加子进度条）
        disturb_batch = []
        disturb_codes = group['Disturb Code'].tolist()
        # for disturb_code in group['Disturb Code']:
        for disturb_code in tqdm(disturb_codes,
                                 desc=f"Rank {rank} Samples",
                                 leave=False,  # 不保留子进度条
                                 unit="sample"):
            disturb_feat = extract_features(rank, disturb_code, true_label, tokenizer, model, args)
            if disturb_feat is not None and disturb_feat.ndim == 1:
                disturb_batch.append(disturb_feat)

        # 存储有效数据
        if disturb_batch:
            n = len(disturb_batch)
            ranks.extend([rank] * n)
            clean_feats.extend([clean_feat] * n)
            disturb_feats.extend(disturb_batch)

    # 保存为压缩格式
    if len(ranks) > 0:
        np.savez_compressed(
            './analysis/features_disturbed.npz',
            ranks=np.array(ranks, dtype=np.int32),
            clean_features=np.array(clean_feats, dtype=np.float32),
            disturb_features=np.array(disturb_feats, dtype=np.float32)
        )
        print(f"成功保存{len(ranks)}对特征，维度：{clean_feats[0].shape}")
    else:
        print("无有效数据保存")


if __name__ == '__main__':
    main()
