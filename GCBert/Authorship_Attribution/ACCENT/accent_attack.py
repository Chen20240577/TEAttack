# -*- coding: utf-8 -*-
# python accent_attack.py
"""
Main script for ACCENT-style identifier-replacement attack (adapted to classification).
参考 wir_attack.py 的结构：加载模型、数据、nearest_k，循环调用攻击器保存对抗样本。
"""

import json
import os
import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

import logging
import argparse
import warnings
import torch
import time

from run_parser import get_identifiers, get_example
from model import Model
from utils import set_seed, build_vocab
from utils import Recorder, Disturb_Recorder
from run import TextDataset
from accent_attacker import Accent_Attacker

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="Evaluation data file (text file).")
    parser.add_argument("--nearest_k_path", default=None, type=str,
                        help="Path to pickle containing nearest_k candidates per example.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model architecture.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Tokenizer name or path")
    parser.add_argument("--number_labels", type=int, default=2,
                        help="Number of labels")
    parser.add_argument("--adv_store_path", type=str,
                        help="Path to store the adv CSV file")
    parser.add_argument("--disturb_store_path", type=str,
                        help="Path to store the disturb CSV file")
    parser.add_argument("--language_type", type=str, default='python',
                        help="Programming language type")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Cache dir")
    # parser.add_argument("--block_size", default=-1, type=int,
    #                     help="Max sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    # Load tokenizer & model config
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_name_or_path:
        encoder = model_class.from_pretrained(args.model_name_or_path,
                                              from_tf=bool('.ckpt' in args.model_name_or_path),
                                              config=config,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        encoder = model_class(config)

    model = Model(encoder, config, tokenizer, args)

    checkpoint_prefix = 'best_acc_model/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    logger.info("reload model from {}".format(output_dir))

    # Load Dataset (TextDataset reads tokenized test file; re-use your run.TextDataset)
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # Build code list from companion *_subs.jsonl if present (same as wir)
    file_type = args.eval_data_file.split('/')[-1].split('.')[0]  # valid
    folder = '/'.join(args.eval_data_file.split('/')[:-1])
    codes_file_path = os.path.join(folder, '{}_subs.jsonl'.format(file_type))
    print("codes_file_path:", codes_file_path)
    source_codes = []

    with open(codes_file_path) as rf:
        for line in rf:
            item = json.loads(line.strip())
            source_codes.append(item["code"].replace("\\n", "\n").replace('\"', '"'))
    assert (len(source_codes) == len(eval_dataset))

    # Build code tokens vocab (used if needed for candidate generation)
    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code, args.language_type)[1])
    id2token, token2id = build_vocab(code_tokens, 5000)

    # Load nearest_k candidates per example (pickle); expected format: list or dict per index:
    # e.g. list_of_nearest_k[index] = { 'var1': ['cand1','cand2',...], 'var2': [...], ... }
    nearest_k_map = None
    if args.nearest_k_path is not None:
        import pickle
        with open(args.nearest_k_path, 'rb') as pf:
            nearest_k_map = pickle.load(pf)
        print("Loaded nearest_k from", args.nearest_k_path)
    else:
        print("Warning: no nearest_k_path provided; substitutes will be sampled from id2token.")

    success_attack = 0
    total_cnt = 0

    adv_record = Recorder(args.adv_store_path)
    disturb_recoder = Disturb_Recorder(args.disturb_store_path)

    attacker = Accent_Attacker(args, model, tokenizer, token2id, id2token, nearest_k_map)
    start_time = time.time()
    query_times = 0

    for index, example in enumerate(eval_dataset):
        example_start_time = time.time()
        code = source_codes[index]

        # get model predictions on original example
        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        true_label = example[3].item()
        # skip if original predicted wrong
        if not orig_label == true_label:
            continue

        identifiers, orig_code_tokens = get_identifiers(code, args.language_type)
        identifiers = [iden[0] for iden in identifiers]
        if len(identifiers) == 0:
            continue

        total_cnt += 1
        print("Index {}: identifiers: {}".format(index, identifiers))

        (code, prog_length, adv_code, true_label, orig_label, temp_label,
         is_success, variable_names, names_to_importance_score, nb_changed_var,
         nb_changed_pos, replaced_words) = attacker.attack_example(example, code, index, disturb_recoder)

        example_end_time = (time.time() - example_start_time) / 60
        print("\nIndex: ", index)
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','

        if is_success == 1:
            success_attack += 1
            adv_record.write(index, code, prog_length, adv_code,
                             true_label, orig_label, temp_label, is_success,
                             variable_names, names_to_importance_score,
                             nb_changed_var, nb_changed_pos,
                             replace_info, "ACCENT", model.query - query_times, example_end_time)
        else:
            adv_record.write(index, code, prog_length, None,
                             true_label, orig_label, temp_label, is_success,
                             variable_names, names_to_importance_score,
                             nb_changed_var, nb_changed_pos,
                             replace_info, "ACCENT", model.query - query_times, example_end_time)

        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()
