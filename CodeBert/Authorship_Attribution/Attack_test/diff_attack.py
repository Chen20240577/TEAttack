# -*- coding: utf-8 -*-

# coding=utf-8
# @Time    : 2020/8/25
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : attacker.py
'''For attacking CodeBERT models'''
import os
import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

retval = os.getcwd()

import json
import logging
import argparse
import warnings
import torch
import time
from run import set_seed
from run import TextDataset
from run import InputFeatures
from utils import Recorder, Disturb_Recorder
from model import Model
from run_parser import get_identifiers
from record_attacker import Attacker

from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning\

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

logger = logging.getLogger(__name__)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_example_embedding(example, model, tokenizer, device):
    """从example对象中提取特征向量"""
    # 从example中获取input_ids，这是已经tokenize和padding好的
    input_ids = example.input_ids
    # 转换为tensor并添加batch维度
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        # 获取模型的隐藏状态
        outputs = model.encoder(input_tensor)
        # 取[CLS]位置的向量作为整个序列的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
        return cls_embedding.squeeze(0).cpu().numpy()


def compute_similarity_from_examples(examples, model, tokenizer, device):
    """从example列表中计算相似度矩阵"""
    n_samples = len(examples)
    print(f"从 {n_samples} 个example中提取特征向量...")

    embeddings = []
    for idx, example in enumerate(examples):
        emb = get_example_embedding(example, model, tokenizer, device)
        embeddings.append(emb)

        if (idx + 1) % 20 == 0:
            print(f"  已处理 {idx + 1}/{n_samples} 个样本")

    embeddings_array = np.vstack(embeddings)
    print("计算余弦相似度矩阵...")
    sim_matrix = cosine_similarity(embeddings_array)
    np.fill_diagonal(sim_matrix, 0)  # 将对角线设为0
    print("相似度矩阵计算完成。")
    return sim_matrix


import random
import math
def simulated_annealing_optimize_min(sim_matrix, iterations=10000, temp=1.0, cooling_rate=0.995):
    """使用模拟退火算法优化样本顺序，最大化相邻相似度之和。
    Args:
        sim_matrix: 相似度矩阵，sim_matrix[i][j] 表示样本i和j的相似度。
        iterations: 迭代次数。
        temp: 初始温度。
        cooling_rate: 降温系数。
    Returns:
        best_order: 最优顺序的索引列表。
        best_score: 该顺序对应的相邻相似度总和。
    """
    n = len(sim_matrix)
    # 初始解：随机排列
    current_order = list(range(n))
    random.shuffle(current_order)
    current_score = sum(sim_matrix[current_order[i]][current_order[i + 1]] for i in range(n - 1))

    best_order = current_order.copy()
    best_score = current_score

    for i in range(iterations):
        # 生成邻域解：随机交换两个位置
        new_order = current_order.copy()
        a, b = random.sample(range(n), 2)
        new_order[a], new_order[b] = new_order[b], new_order[a]

        # 计算新解分数（只计算变化的部分，提高效率）
        new_score = current_score
        # 减去旧连接的影响
        for pos in {a - 1, a}:
            if 0 <= pos < n - 1:
                new_score -= sim_matrix[current_order[pos]][current_order[pos + 1]]
        # 加上新连接的影响
        for pos in {a - 1, a}:
            if 0 <= pos < n - 1:
                new_score += sim_matrix[new_order[pos]][new_order[pos + 1]]
        for pos in {b - 1, b}:
            if 0 <= pos < n - 1 and pos not in {a - 1, a}:  # 避免重复计算
                new_score -= sim_matrix[current_order[pos]][current_order[pos + 1]]
                new_score += sim_matrix[new_order[pos]][new_order[pos + 1]]

        # 决定是否接受新解（接受分数更低的解）
        if new_score < current_score or random.random() < math.exp((current_score - new_score) / temp):
            current_order, current_score = new_order, new_score

        # 更新全局最优解（改为记录最低分数）
        if current_score < best_score:
            best_order, best_score = current_order.copy(), current_score

        # 降温
        temp *= cooling_rate

        if (i + 1) % 1000 == 0:
            print(f"  模拟退火迭代 {i + 1}/{iterations}, 当前分数: {current_score:.4f}, 最优分数: {best_score:.4f}")

    print(f"模拟退火完成。最低相邻相似度总和: {best_score:.4f}")
    return best_order, best_score

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--adv_store_path", type=str,
                        help="Path to store the adv CSV file")
    parser.add_argument("--disturb_store_path", type=str,
                        help="Path to store the disturb CSV file")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--language_type", type=str,
                        help="The programming language type of dataset")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        # model2 = model_class.from_pretrained(args.model_name_or_path,
        #                                     from_tf=bool('.ckpt' in args.model_name_or_path),
        #                                     config=config,
        #                                     cache_dir=args.cache_dir if args.cache_dir else None)
        # model2的思路无效，但是保留当个记录
    else:
        model = model_class(config)
        # model2 = model_class(config)

    model = Model(model, config, tokenizer, args)
    # model2 = Model(model2, config, tokenizer, args)


    checkpoint_prefix = 'best_acc_model/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    # model2.load_state_dict(torch.load(output_dir))
    # model2.to(args.device)
    # model2.eval()
    # model.eval()  # <--- 添加评估模式
    # 可以作为一个小的发现提进去，在评估的时候，模型更容易受到攻击

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')
    # codebert_mlm.eval()  # <--- 添加评估模式

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    file_type = args.eval_data_file.split('/')[-1].split('.')[0]  # valid
    folder = '/'.join(args.eval_data_file.split('/')[:-1])  # 得到文件目录
    codes_file_path = os.path.join(folder, '{}_subs.jsonl'.format(
        file_type))
    print(codes_file_path)
    source_codes = []
    substs = []
    with open(codes_file_path) as rf:
        for line in rf:
            item = json.loads(line.strip())
            source_codes.append(item["code"].replace("\\n", "\n").replace('\"', '"'))
            substs.append(item["substitutes"])
    assert (len(source_codes) == len(eval_dataset) == len(substs))
    # print(len(source_codes) , len(eval_dataset) , len(substs))

    # 现在要尝试计算importance_score了.
    success_attack = 0
    total_cnt = 0

    adv_record = Recorder(args.adv_store_path)
    disturb_recoder = Disturb_Recorder(args.disturb_store_path)

    # attacker = Attacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0, model2 = model2)
    attacker = Attacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0

    # torch.set_grad_enabled(False)
    # target=[0, 1, 16, 18, 34, 35, 40, 41, 55, 56, 60, 72, 82, 95, 105, 110, 122]
    # target = [16, 18, 34, 35, 40, 41, 55, 56, 60, 72, 82, 95, 105, 110, 122]

    print("开始基于example特征优化样本处理顺序...")

    # 1. 直接从TextDataset的examples属性中获取所有example
    examples_list = eval_dataset.examples
    # 验证长度一致
    assert len(examples_list) == len(source_codes) == len(substs)

    # 2. 计算相似度矩阵
    sim_matrix = compute_similarity_from_examples(examples_list, model, tokenizer, args.device)

    # 3. 使用优化算法得到新的处理顺序
    new_order_indices, total_similarity = simulated_annealing_optimize_min(sim_matrix, iterations=5000)
    print(f"优化完成。新顺序的相邻样本相似度总和为: {total_similarity:.4f}")

    # 4. 按照新顺序重新组织数据
    # 注意：这里我们重新排列了eval_dataset.examples本身
    eval_dataset.examples = [examples_list[i] for i in new_order_indices]
    source_codes = [source_codes[i] for i in new_order_indices]
    substs = [substs[i] for i in new_order_indices]


    for index, example in enumerate(eval_dataset):
        # if index not in target:
        #     continue
        original_index = new_order_indices[index]
        example_start_time = time.time()
        code = source_codes[index]
        subs = substs[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(
            example, code, subs, index, disturb_recoder)

        attack_type = "Greedy"
        if is_success == -1 and args.use_ga:
            # 如果不成功，则使用gi_attack
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
                example, code, subs, index, disturb_recoder, initial_replace=replaced_words)
            attack_type = "GA"

        example_end_time = (time.time() - example_start_time) / 60

        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)
        adv_record.write(original_index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success,
                         variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type,
                         model.query - query_times, example_end_time)
        query_times = model.query

        if is_success >= -1:
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1

        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()


if __name__ == '__main__':
    main()
