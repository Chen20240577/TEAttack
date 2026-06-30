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
from utils import Recorder, build_vocab
from attacker import Attacker, compute_cosine_similarity
import math

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

import csv
from tqdm import tqdm
def compute_avg_similarity_from_csv(csv_path, tfidf_vectorizer):
    similarities = []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        print(f"成功读取CSV文件，共有 {len(rows)} 行数据。")

        for row in tqdm(rows, desc="从CSV计算相似度"):
            index_str = row.get('Index', '').strip()
            adv_code = row.get('Adversarial Code', '').strip()
            orig_code = row.get('Original Code', '').strip()

            if adv_code and orig_code:  # 增加对原始代码的检查
                try:
                    similarity, _ = compute_cosine_similarity(orig_code, adv_code, tfidf_vectorizer)
                    similarities.append(similarity)
                except Exception as e:
                    print(f"警告: 处理索引 {index_str} 时计算相似度出错: {e}，跳过。")
                    continue

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            print(f"从 {len(similarities)} 个成功对抗样本中计算出平均相似度。")
            return avg_sim
        else:
            print("警告: CSV中没有有效的成功对抗样本可用于计算相似度。")
            return 0.0

    except FileNotFoundError:
        print(f"错误: 文件 {csv_path} 未找到。")
        return 0.0
    except Exception as e:
        print(f"读取或处理CSV文件 {csv_path} 时发生错误: {e}")
        return 0.0

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

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--block_size', type=int, default=512,
                        help="block size")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=384, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--nearest_k_path", default=None, type=str,
                        help="Path to pickle containing nearest_k candidates per example.")
    parser.add_argument("--adv_store_path", type=str,
                        help="Path to store the adv CSV file")
    parser.add_argument("--reverse", default=False, type=bool,
                        help="Reverse Order Selection of Substitution Words")
    parser.add_argument("--attacker", default="Alert_GA_random_thld_100", type=str,
                        help="Attacker")
    parser.add_argument("--attack_type", default="Alert_GA_thld", type=str,
                        help="Attack type")
    parser.add_argument("--subs", default="random", type=str,
                        help="Substitution Words Kind")
    # subs = random/word2vec/wordnet/model
    parser.add_argument('--K', type=int, default=30,
                        help="The number of top K candidates per example.")
    parser.add_argument('--thld', type=float, default=0.1,
                        help="The threshold")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device
    set_seed(args.seed)

    model_name = load.model_name_get(args.model_type)
    task_name = load.task_name_get(args.task)
    output_dir = os.path.join('../../../', model_name, task_name)

    model, tokenizer, args = load.load_model(args, output_dir)

    dataset = load.load_dataset(args, tokenizer)
    source_codes, substs = load.load_code(args)
    assert (len(source_codes) == len(dataset))

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_1 = code
        code_2 = None

        if args.task == 'clone':
            code_1 = deepcopy(code[2])
            code_2 = deepcopy(code[3])
        elif args.task == 'authorship' or args.task == 'defect':
            pass
        code_tokens.append(get_identifiers(code_1, args.language_type)[1])
    id2token, token2id = build_vocab(code_tokens, 5000)

    nearest_k_map = None
    if args.nearest_k_path is not None:
        import pickle

        with open(args.nearest_k_path, 'rb') as pf:
            nearest_k_map = pickle.load(pf)
        print("Loaded nearest_k from", args.nearest_k_path)
    else:
        print("Warning: no nearest_k_path provided;")

    adv_record = Recorder(args.adv_store_path)

    tfidf = load.load_tfidf(args.task)

    # 步骤1：判断是否需要进行基于 CSV 的预分析
    pre_attack_similarities = []
    avg_similarity = 0.0
    pre_analysis_csv_path = None

    # 判断攻击者是否包含"thld"
    if hasattr(args, 'attacker') and args.attacker is not None and 'thld' in args.attacker:
        if args.attacker.endswith('thld_000'):
            print(f"\n当前攻击者为 {args.attacker}，这是基础阈值攻击，跳过CSV预分析流程。")
            avg_similarity = 0.0
        else:
            print(f"\n=== 检测到攻击者包含 'thld' ({args.attacker})，启动CSV优先的预分析流程 ===")
            # 步骤2：构建用于预分析的 CSV 文件路径（文件名调整）
            # 规则：将攻击者字符串中 'thld_' 之后的部分替换为 '000'
            import re
            import os

            if args.adv_store_path:
                original_dir = os.path.dirname(args.adv_store_path)
                original_filename = os.path.basename(args.adv_store_path)
                # 从文件名中分离出原始攻击者和模型名（这部分取决于您的实际命名规则，以下是通用化示例）
                # 例如: "TEAA_greedy_bert.csv" -> 攻击者: TEAA_greedy, 模型: bert
                # 由于您指定"csv的文件起名和主攻击流程一致，但是微调一下"，这里我们直接替换攻击者部分
                new_filename = re.sub(r'(thld_)(\d+)', r'\g<1>000', original_filename)

                # 如果替换没有发生（比如文件名是 MHM_thld_Bert.csv 没有数字部分），
                # 则在 thld_ 后面插入 000
                if new_filename == original_filename and 'thld_' in new_filename:
                    # 在 thld_ 后面插入 000
                    new_filename = re.sub(r'(thld_)', r'\g<1>000_', new_filename)
                    # 移除可能多余的下划线
                    new_filename = new_filename.replace('thld_000_', 'thld_000')

                pre_analysis_csv_path = os.path.join(original_dir, new_filename)
            else:
                print("警告: args.adv_store_path 未设置，无法生成预分析CSV路径。")
                pre_analysis_csv_path = None

            # 步骤3：检查预分析的 CSV 文件是否存在
            if pre_analysis_csv_path and os.path.exists(pre_analysis_csv_path):
                print(f"找到已存在的预分析CSV文件: {pre_analysis_csv_path}，直接读取并计算平均相似度...")
                # 调用函数（下面定义）来读取CSV并计算平均相似度
                avg_similarity = compute_avg_similarity_from_csv(pre_analysis_csv_path, tfidf)
                print(f"从CSV计算得到的平均余弦相似度: {avg_similarity:.4f}")
            else:
                print(f"预分析CSV文件不存在({pre_analysis_csv_path})。")
    else:
        print(f"\n攻击者({args.attacker})不包含 'thld'，跳过CSV优先的预分析流程。")

    # record = importance_recoder(args.importance_store_path)
    attacker = Attacker(args, model, tokenizer, avg_similarity, tfidf, nearest_k_map, id2token, substs)

    success_attack = 0
    total_cnt = 0
    start_time = time.time()
    query_times = 0

    for index, example in enumerate(dataset):
        example_start_time = time.time()

        code = source_codes[index]

        logits, preds = model.get_results([example], args.batch_size)
        orig_label = preds[0]

        true_label = load.label_get(example, args.model_type, args.task)

        if not orig_label == true_label:
            continue

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

        total_cnt += 1
        print("\nIndex {}: identifiers: {}".format(index, identifiers))

        if args.attack_type == "TEAA_greedy":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.TEAA_greedy_attack(index, example, code_1, true_label, code_2)
        elif args.attack_type == "TEAA_broad":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.TEAA_broad_attack(index, example, code_1, true_label, code_2)
        elif args.attack_type == "TEAA_wordnet":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.TEAA_greedy_attack(index, example, code_1, true_label, code_2)
        elif args.attack_type == "TEAA_subs":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.TEAA_subs_attack(index, example, code_1, true_label, code_2,
                                                                         substs[index])
        elif args.attack_type == "TEAA_num":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.TEAA_num_attack(index, example, code_1, true_label, code_2)
        # 前面是探索方法时的代码，TEAA仅标注，但是没有实际使用这个框架

        elif args.attack_type == "MHM_thld":
            _res = attacker.MHM_thld_attack(code_1, true_label, code_2, id2token,_n_candi=args.K, subs=substs[index])
            # 虽然传入了subs，但是只用到了目标替换词，没有实际调用里面的替换词，替换词的来源还是随机生成的
            prog_length = _res["prog_length"]
            adv_code = _res['tokens']
            temp_label = _res["new_pred"]
            is_success = _res["is_success"]
            variable_names = _res["old_uid"]
            names_to_importance_score = _res["score_info"]
            nb_changed_var = _res["nb_changed_var"]
            nb_changed_pos = _res["nb_changed_pos"]
            replaced_words = _res["replace_info"]

        elif args.attack_type == "Alert_GA_thld":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.Alert_GA_thld(index, example, code_1, true_label, code_2)
        elif args.attack_type == "Alert_GR_thld":
            (code, prog_length, adv_code, true_label, orig_label, temp_label,
             is_success, variable_names, names_to_importance_score, nb_changed_var,
             nb_changed_pos, replaced_words) = attacker.Alert_GR_thld(index, example, code_1, true_label, code_2)


        example_end_time = (time.time() - example_start_time) / 60
        print("Index: ", index)
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
                             replace_info, args.attack_type, model.query - query_times, example_end_time)
        else:
            adv_record.write(index, code, prog_length, None,
                             true_label, orig_label, temp_label, is_success,
                             variable_names, names_to_importance_score,
                             nb_changed_var, nb_changed_pos,
                             replace_info, args.attack_type, model.query - query_times, example_end_time)
        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))

    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
