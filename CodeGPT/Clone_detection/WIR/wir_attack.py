import os
import sys
from copy import deepcopy

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

retval = os.getcwd()

import logging
import argparse
import warnings
import pickle
import torch
import time
from run_parser import get_identifiers, get_example
from model import Model
from utils import set_seed, build_vocab
from utils import Recorder, Disturb_Recorder
from run import TextDataset
from wir_attacker import WIR_Attacker
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer)
}

logger = logging.getLogger(__name__)


def get_code_pairs(file_path):
    postfix = file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])  # 得到文件目录
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
        postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")
    # parser.add_argument("--csv_store_path", default=None, type=str,
    #                     help="Base Model")
    parser.add_argument("--adv_store_path", type=str,
                        help="Path to store the adv CSV file")
    parser.add_argument("--disturb_store_path", type=str,
                        help="Path to store the disturb CSV file")
    parser.add_argument("--language_type", type=str,
                        help="The programming language type of dataset")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 用结束符作为填充符
    config.pad_token_id = tokenizer.pad_token_id

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)

    checkpoint_prefix = 'best_acc_model/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    logger.info("reload model from {}".format(output_dir))

    # Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    # Load code pairs
    source_codes = get_code_pairs(args.eval_data_file)

    assert len(source_codes) == len(eval_dataset)

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code[2], args.language_type)[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    success_attack = 0
    total_cnt = 0

    adv_record = Recorder(args.adv_store_path)
    disturb_recoder = Disturb_Recorder(args.disturb_store_path)

    attacker = WIR_Attacker(args, model, tokenizer, token2id, id2token)
    start_time = time.time()
    query_times = 0

    for index, example in enumerate(eval_dataset):

        example_start_time = time.time()
        code_pair = source_codes[index]

        logits, preds = model.get_results([example], args.eval_batch_size)
        orig_label = preds[0]
        true_label = example[1].item()
        if not orig_label == true_label:
            continue
        code_1 = deepcopy(code_pair[2])

        code_2 = deepcopy(code_pair[3])
        code_2 = " ".join(code_2.split())

        identifiers, orig_code_tokens = get_identifiers(code_1, args.language_type)
        identifiers = [iden[0] for iden in identifiers]
        if len(identifiers) == 0:
            continue
        total_cnt += 1
        print("identifier:", identifiers)

        code_pair, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = (
            attacker.wir_random_attack(example, code_pair, code_1, code_2, index, disturb_recoder))

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
            adv_record.write(index, code_pair, prog_length, adv_code,
                             true_label, orig_label, temp_label, is_success,
                             variable_names, names_to_importance_score,
                             nb_changed_var, nb_changed_pos,
                             replace_info, "WIR",
                             model.query - query_times, example_end_time)

        else:
            adv_record.write(index, code_pair, prog_length, None,
                             true_label, orig_label, temp_label, is_success,
                             variable_names, names_to_importance_score,
                             nb_changed_var, nb_changed_pos,
                             replace_info, "WIR",
                             model.query - query_times, example_end_time)

        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()
