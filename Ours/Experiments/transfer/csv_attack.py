import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import random
import warnings
import argparse

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')

from load import load
from run_parser import get_identifiers, get_example
from utils import Disturb_Recorder
from csv_attacker import Attacker

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
    parser.add_argument("--csv_read_path", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--transfer_store_path", type=str,
                        help="Path to store the transfer CSV file")

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

    model_name = load.model_name_get(args.model_type)
    task_name = load.task_name_get(args.task)
    output_dir = os.path.join('../../../', model_name, task_name)

    model, tokenizer, args = load.load_model(args, output_dir)
    model.eval()

    # # ========= 恢复 RNG =========
    # attack_model_dir = os.path.join(output_dir, '{}'.format("saved_attack_models/idx_0/rng_state.pt"))
    # rng_state = torch.load(attack_model_dir)
    # torch.set_rng_state(rng_state["torch_rng"])
    # if torch.cuda.is_available() and rng_state["cuda_rng"] is not None:
    #     torch.cuda.set_rng_state(rng_state["cuda_rng"])
    # np.random.set_state(rng_state["numpy_rng"])
    # random.setstate(rng_state["python_rng"])
    # print("RNG restored ✔")

    transfer_recoder = Disturb_Recorder(args.transfer_store_path)
    attacker = Attacker(args, model, tokenizer)
    attacker.csv_attack(args.csv_read_path, args, transfer_recoder)
