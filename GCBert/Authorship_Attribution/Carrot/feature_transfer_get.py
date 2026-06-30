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

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG import DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)

from tree_sitter import Language, Parser

from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
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


def conduct_tokens(code, tokenizer, args):
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
    return source_tokens


def conduct_input_ids(code, tokenizer, args):
    # 构造示例（假设示例格式与之前相同）
    tokens = conduct_tokens(code, tokenizer, args)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return torch.tensor(input_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_features(code, tokenizer, model, args):
    try:
        # Generate model input
        input_ids = conduct_input_ids(code, tokenizer, args)

        # Prepare input for model
        input_ids = input_ids.unsqueeze(0).to(args.device)  # Add batch dimension
        attention_mask = (input_ids != tokenizer.pad_token_id).float().to(args.device)

        # Extract features from the model
        with torch.no_grad():
            outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get the last hidden state (batch_size, seq_length, hidden_size)
            hidden_states = outputs.hidden_states[-1]

            # Take CLS token embedding (batch_size, hidden_size)
            cls_embedding = hidden_states[:, 0, :]

            # Detach and convert to numpy
            return cls_embedding.squeeze(0).cpu().numpy()

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
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="模型类型")
    parser.add_argument("--config_name", default="../../../../../Models/microsoft/graphcodebert-base", type=str,
                        help="配置文件路径")
    parser.add_argument("--model_name_or_path", default="../../../../../Models/microsoft/graphcodebert-base", type=str,
                        help="模型路径")
    parser.add_argument("--tokenizer_name", default="../../../../../Models/microsoft/graphcodebert-base", type=str,
                        help="tokenizer路径")

    # 训练/评估参数
    parser.add_argument("--code_length", default=384, type=int,
                        help="输入序列最大长度")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="输入序列最大长度")

    parser.add_argument("--num_labels", default=66, type=int,
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
        clean_feat = extract_features(original_code, tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue

        # 处理Disturb Code样本
        dist_code = group['Disturb Code'].iloc[0]
        dist_feat = extract_features(dist_code, tokenizer, model, args)
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
