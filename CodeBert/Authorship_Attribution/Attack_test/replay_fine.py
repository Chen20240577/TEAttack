# -*- coding: utf-8 -*-

import os
import json
import sys

import torch
import numpy as np
import random

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')
sys.path.append('../../../Ours/Experiments')
from load import load

from model import Model
from run import InputFeatures
from utils import CodeDataset
from record_attacker import convert_code_to_features

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


# ========= 路径 =========
attack_model_dir = "../saved_attack_models/idx_0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= 加载输入 =========
with open(os.path.join(attack_model_dir, "input.json")) as f:
    data = json.load(f)

clean_code = data["orig_code"]
adv_code = data["adv_code"]
true_label = data["true_label"]

print("Loaded input ✔")


class Args:
    pass

args = Args()
args.model_type = "bert"
args.number_labels = 66
args.task = 'authorship'
args.checkpoint_prefix = "saved_models/best_acc_model/model.bin"
args.block_size = 512
args.eval_batch_size = 1
args.device = device

model_name = load.model_name_get(args.model_type)
task_name = load.task_name_get(args.task)
output_dir = os.path.join('../../../', model_name, task_name)
model, tokenizer, args = load.load_model(args, output_dir)
model.eval()

print("Model loaded ✔")


# ========= 构造输入 =========
feat = load.conduct_example(
    args,
    clean_code,
    true_label,
    tokenizer,
    args.block_size,
    tokenizer.cls_token,
    tokenizer.sep_token,
    None
)

data = CodeDataset([feat])
# ========= 单次验证 =========
logit, pred = model.get_results(data, 1)

print("\n===== 原始预测 =====")
print("Pred:", pred[0])
print("Logits:", logit[0])


feature = load.conduct_example(
    args,
    clean_code,
    true_label,
    tokenizer,
    args.block_size,
    tokenizer.cls_token,
    tokenizer.sep_token,
    adv_code
)

dataset = CodeDataset([feature])


# ========= 验证 =========
logits, preds = model.get_results(dataset, 1)

print("\n===== load =====")
print("Pred:", preds[0])
print("Logits:", logits[0])


feature = convert_code_to_features(
    adv_code,
    tokenizer,
    true_label,
    args
)

dataset = CodeDataset([feature])


# ========= 验证 =========
logits, preds = model.get_results(dataset, 1)

print("\n===== convert_code_to_features =====")
print("Pred:", preds[0])
print("Logits:", logits[0])