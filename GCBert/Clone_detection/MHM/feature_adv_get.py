# -*- coding: utf-8 -*-
import ast
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

from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)
from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG import DFG_ruby, DFG_go, DFG_php, DFG_javascript
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)


def extract_dataflow(code, parser, lang):
    # remove comments
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


def conduct_tokens(code, parser, tokenizer, args):
    # extract data flow
    code_tokens, dfg = extract_dataflow(code, parser, 'java')
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    # truncating
    code_tokens = code_tokens[
        :args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][
        :512 - 3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]

    return source_tokens


def conduct_input_ids(code, tokenizer, args):
    parser = parsers['java']
    tokens = conduct_tokens(code, parser, tokenizer, args)
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
        # attention_mask = attn_mask

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


class Args:
    def __init__(self):
        self.mlm = False
        self.mlm_probability = 0.15
        self.code_length = 384
        self.data_flow_length = 128
        self.device = torch.device("cuda")
        self.num_labels = 2
        self.do_lower_case = False
        self.cache_dir = None
        self.output_dir = "../saved_models"
        self.model_type = "roberta"
        self.config_name = "../../../../../Models/microsoft/graphcodebert-base"
        self.model_name_or_path = "../../../../../Models/microsoft/graphcodebert-base"
        self.tokenizer_name = "../../../../../Models/microsoft/graphcodebert-base"
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
    df = pd.read_csv('../../../AdvExamples/Clone_detection/MHM_GCBert.csv').dropna()
    df['Original Code'] = df['Original Code'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    df = df.astype({
        'Index': int,
        'Original Code': object,  # 使用object类型存储列表
        'Adversarial Code': str,
        'True Label': int
    })

    # 存储结构初始化
    ranks = []
    clean_feats = []
    adv_feats = []

    # 按Index分组处理（主进度条）
    for rank, group in tqdm(df.groupby('Index'), desc="Processing Groups"):
        # 数据一致性验证 - 使用字符串表示进行比较
        original_codes = group['Original Code'].apply(str).unique()
        true_labels = group['True Label'].unique()

        if len(original_codes) != 1 or len(true_labels) != 1:
            print(f"跳过不一致的Index {rank}")
            continue

        original_code = group['Original Code'].iloc[0]

        # 提取干净样本特征
        clean_feat = extract_features(original_code[2], tokenizer, model, args)
        if clean_feat is None or clean_feat.ndim != 1:
            continue
        # 直接处理单个对抗样本
        adv_code = group['Adversarial Code'].iloc[0]
        adv_feat = extract_features(adv_code, tokenizer, model, args)
        if adv_feat is None or adv_feat.ndim != 1:
            continue

        # 存储特征对
        ranks.append(rank)
        clean_feats.append(clean_feat)
        adv_feats.append(adv_feat)

        # print(f"处理Index {rank}: "
        #       f"URL1类型={type(original_code[0])}, "
        #       f"代码1类型={type(original_code[2])}, "
        #       f"对抗代码类型={type(adv_code)}")

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
