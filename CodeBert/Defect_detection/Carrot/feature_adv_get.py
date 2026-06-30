# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append('../../')
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

from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)


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
        input_ids, _ = conduct_example(rank, code, int(label), tokenizer, args)

        # 调整输入形状
        input_ids = input_ids.view(-1, args.block_size).to(args.device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # 使用模型的基础编码器获取隐藏状态
        with torch.no_grad():
            # 关键修改：直接访问基础模型（roberta）而非分类头
            outputs = model.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True)

            # 获取最后一个隐藏层状态
            last_hidden_states = outputs.hidden_states[-1]

            # 提取[CLS]标记的特征
            cls_feature = last_hidden_states[:, 0, :].cpu().numpy().squeeze()
            return cls_feature

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
        self.model_type = "roberta"
        self.config_name = "../../../../../Models/microsoft/codebert-base"
        self.model_name_or_path = "../../../../../Models/microsoft/codebert-base"
        self.tokenizer_name = "../../../../../Models/FacebookAI/roberta-base"
        self.seed = 123456


def main():
    # 创建模拟的args对象
    args = Args()

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
    df = pd.read_csv('../../../AdvExamples/Defect_detection/Carrot_Bert.csv').dropna()
    df = df.astype({'Index': int, 'Original Code': str, 'Adversarial Code': str, 'True Label': int})

    # 存储结构初始化
    ranks = []
    clean_feats = []
    adv_feats = []

    # 按Index分组处理（主进度条）
    for rank, group in tqdm(df.groupby('Index'), desc="Processing Groups"):
        # 数据一致性验证
        if len(group['Original Code'].unique()) != 1 or len(group['True Label'].unique()) != 1:
            print(f"跳过不一致的Index {rank}")
            continue

        original_code = group['Original Code'].iloc[0]
        true_label = group['True Label'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(rank, original_code, true_label, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        # 直接处理单个对抗样本
        adv_code = group['Adversarial Code'].iloc[0]
        adv_feat = extract_features(rank, adv_code, true_label, tokenizer, model, args)
        if adv_feat is None or adv_feat.ndim != 1:
            continue

        # 存储特征对
        ranks.append(rank)
        clean_feats.append(clean_feat)
        adv_feats.append(adv_feat)

    # 保存为压缩格式
    if len(ranks) > 0:
        np.savez_compressed(
            './analysis/features_adv.npz',
            ranks=np.array(ranks, dtype=np.int32),
            clean_features=np.array(clean_feats, dtype=np.float32),
            adv_features=np.array(adv_feats, dtype=np.float32)
        )
        print(f"成功保存{len(ranks)}对特征，维度：{clean_feats[0].shape}")
    else:
        print("无有效数据保存")


if __name__ == '__main__':
    main()
