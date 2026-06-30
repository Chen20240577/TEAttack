import os
import sys
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import random
import warnings
import time
import argparse

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')

from load import load

from run_parser import get_identifiers, get_example
from utils import importance_recoder
from scanner import Scanner

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # task, model_type, output_dir, checkpoint_prefix, device
    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")

    parser.add_argument("--task", default=None, type=str, required=True,
                        help="The task.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--checkpoint_prefix", default='best_acc_model/model.bin', type=str,
                        help="The model checkpoint for weights initialization.")

    ## Other parameters
    parser.add_argument("--language_type", type=str,
                        help="The programming language type of dataset")
    parser.add_argument("--number_labels", type=int,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="batch size")

    parser.add_argument("--importance_store_path", type=str,
                        help="Path to store the importance CSV file")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--block_size', type=int, default=512,
                        help="block size")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=384, type=int,
                        help="Optional Code input sequence length after tokenization.")


    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device
    set_seed(args.seed)

    model_name = None
    task_name = None

    if args.model_type == "bert":
        model_name = "CodeBert"
    elif args.model_type == "gcbert":
        model_name = "GCBert"
    elif args.model_type == "t5":
        model_name = "CodeT5"
    elif args.model_type == "gpt2":
        model_name = "CodeGPT"

    if args.task == "authorship":
        task_name = "Authorship_Attribution"
    elif args.task == "clone":
        task_name = "Clone_detection"
    elif args.task == "defect":
        task_name = "Defect_detection"

    output_dir = os.path.join('../../../', model_name, task_name)

    # print(args.language_type)
    # print(type(args.language_type))

    model, tokenizer, args = load.load_model(args, output_dir)
    dataset = load.load_dataset(args,tokenizer)
    source_codes,_ = load.load_code(args)
    assert (len(source_codes) == len(dataset))

    record = importance_recoder(args.importance_store_path)
    scan = Scanner(args, model, tokenizer)

    for index, example in enumerate(dataset):
        code = source_codes[index]
        logits, preds = model.get_results([example], args.batch_size)
        orig_label = preds[0]

        true_label = load.label_get(example, args.model_type, args.task)

        if not orig_label == true_label:
            continue
        # 这段代码是试图证明不同模型的敏感标识符位置不一致，所以不用比较是否预测正确
        identifiers = []

        code_1 = code
        code_2 = None

        if args.task == 'clone':
            code_1 = deepcopy(code[2])
            code_2 = deepcopy(code[3])
        elif args.task == 'authorship' or args.task == 'defect':
            pass

        identifiers, orig_code_tokens = get_identifiers(code_1, args.language_type)
        identifiers = [iden[0] for iden in identifiers]
        if len(identifiers) == 0:
            continue
        print("Index {}: identifiers: {}".format(index, identifiers))

        names_positions_dict, names_to_importance_score = scan.scanner(code_1, true_label, code_2, identifiers)
        record.write(index, code, names_positions_dict, names_to_importance_score)























