# -*- coding: utf-8 -*-
"""
@author: einsam
"""

import logging
import sys

sys.path.append('../../../')

import re
import random
import torch
import argparse
import os, time
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from codebert import CodeBERTModel
from tqdm import tqdm
from dataset import VulDataset
from modifier import TokenModifier
from utils import set_seed, Adversarial_Recorder, Disturb_Recorder

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

# 配置日志
logger = logging.getLogger('Attacker')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Attacker(object):
    def __init__(self, dataset, classifier, tokenizer, adv_record, disturb_recoder, device):
        self.tokenizer = tokenizer
        self.txt2idx = dataset.get_txt2idx()
        # self.idx2txt = dataset.get_idx2txt()
        self.adv_record = adv_record
        self.disturb_recoder = disturb_recoder
        self.classifier = classifier
        self.dataset = dataset
        self.device = device
        # 预加载模型信息
        logger.info("Preparing token modifier...")

        self.tokenM = TokenModifier(
            classifier=classifier,
            uids=self._extract_all_identifiers(dataset),
            tokenizer=tokenizer,
            device=device
        )
        self._candidate_cache = {}  # 缓存候选标识符结果

    def _extract_all_identifiers(self, dataset):
        """动态从测试集提取所有标识符，只执行一次"""
        logger.info("Extracting identifiers from dataset...")
        identifiers = set()
        size = dataset.test.get_size()

        # 进度条显示
        pbar = tqdm(total=size, desc="Extracting identifiers")
        for i in range(size):
            batch = dataset.test.next_batch(1)
            tokens = batch['raw'][0]  # 原始token列表
            for token in tokens:
                if token.isidentifier():
                    identifiers.add(token)
            pbar.update(1)
        pbar.close()

        logger.info(f"Extracted {len(identifiers)} unique identifiers")
        return list(identifiers)

    def _filter_identifiers(self, identifiers, code_str, true_label):
        """过滤不太可能影响预测的标识符"""
        filtered = set()
        threshold = 2  # 出现次数阈值

        for ident in identifiers:
            # 跳过过短或过长的标识符
            if len(ident) < 2 or len(ident) > 30:
                continue

            # 跳过常见类型和关键字
            if ident in ['int', 'char', 'void', 'return', 'if', 'else', 'for', 'while']:
                continue

            # 跳过仅出现1次的标识符
            if code_str.count(ident) < threshold:
                continue

            # 跳过在字符串/注释中的标识符
            if f'"{ident}"' in code_str or f"'{ident}'" in code_str:
                continue

            filtered.add(ident)

        logger.debug(f"Filtered identifiers: {len(identifiers)} -> {len(filtered)}")
        return filtered

    def attack(self, x_raw, x_str, y, index, n_candidate=100, n_iter=20):

        true_label = y

        # 预计算原始输入的logits
        with torch.no_grad():
            base_logits = self.classifier.run_batch(x_str)

        base_pred = torch.argmax(base_logits).item()

        # 检查原始预测是否正确
        if base_pred != true_label:
            self.adv_record.write(index, x_str, None, true_label, base_pred, None)
            # 没预测成功的也记录，分析样本的一个小步骤用到了
            logger.info(f"Original mistake. ID={index}, True:{true_label} Pred:{base_pred}")
            return False, base_pred

        # 原始真实类别的logit值（用于跟踪改进）
        base_true_logit = base_logits[0, true_label].item()
        last_improvement = 0  # 跟踪改进次数

        # 使用预编译正则提取所有标识符（更快）
        identifiers = set(re.findall(r'[a-zA-Z_]\w*', x_str))

        # 过滤不太可能影响预测的标识符
        identifiers = self._filter_identifiers(identifiers, x_str, true_label)

        if not identifiers:
            logger.info(f"No valid identifiers to attack. ID={index}")
            return False, true_label

        # 初始化日志记录
        logger.info(
            f"Attack ID={index} with {len(identifiers)} identifiers | True:{true_label} | Init logit:{base_true_logit:.3f}")

        # 主攻击循环
        for iter_idx in range(n_iter):
            logger.debug(f"Attack iteration {iter_idx + 1}/{n_iter} for ID={index}")
            identifiers_list = list(identifiers)
            random.shuffle(identifiers_list)
            iter_success = False

            # 批量收集所有候选
            candidate_replacements = []
            for k in identifiers_list:
                # 跳过无效标识符
                if k not in self.txt2idx or len(k) < 2 or k in ['int', 'char', 'void']:
                    continue

                # 从缓存或生成器获取候选
                cache_key = (index, iter_idx, k)
                if cache_key in self._candidate_cache:
                    new_x_strs, new_x_uid = self._candidate_cache[cache_key]
                else:
                    new_x_strs, new_x_uid = self.tokenM.rename_uid(
                        x_raw, true_label, k, min(n_candidate, 10)  # 限制最大候选数
                    )
                    if new_x_strs:
                        self._candidate_cache[cache_key] = (new_x_strs, new_x_uid)

                if new_x_strs:
                    candidate_replacements.extend(
                        (k, uid, cand_str) for uid, cand_str in zip(new_x_uid, new_x_strs)
                    )

            if not candidate_replacements:
                logger.debug(f"No valid candidates in iteration {iter_idx}")
                continue

            # 批量评估候选
            batch_candidate_strs = [c[2] for c in candidate_replacements]

            # 批量tokenize和模型推理
            batch_preds = []
            batch_true_logits = []

            # 分批处理候选以避免内存溢出
            batch_size = 16  # 根据GPU内存调整
            for i in range(0, len(batch_candidate_strs), batch_size):
                batch_strs = batch_candidate_strs[i:i + batch_size]

                with torch.no_grad():
                    batch_logits = self.classifier.run_batch(batch_strs)

                # 收集结果
                batch_preds.extend(torch.argmax(batch_logits, dim=1).tolist())
                batch_true_logits.extend(batch_logits[:, true_label].tolist())

            best_candidate = None
            best_improvement = 0
            attack_success = False
            success_candidate = None

            for idx, (k, uid, cand_str) in enumerate(candidate_replacements):
                cand_pred = batch_preds[idx]
                true_logit = batch_true_logits[idx]

                # 记录扰动
                if self.disturb_recoder:
                    self.disturb_recoder.write(index, x_str, cand_str, true_label)

                # 成功攻击：预测改变
                if cand_pred != true_label:
                    attack_success = True
                    success_candidate = (k, uid, cand_str, cand_pred)
                    break

                # 跟踪最佳改进（至少降低0.1）
                improvement = base_true_logit - true_logit
                if improvement > 0.1 and improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = (k, uid, cand_str, true_logit)

            # 优先处理攻击成功
            if attack_success and success_candidate:
                k, uid, cand_str, cand_pred = success_candidate
                if self.adv_record:
                    self.adv_record.write(index, x_str, cand_str, true_label, true_label, cand_pred)
                logger.info(f"SUCCESS! ID={index} | {k}=>{uid} | True:{true_label} Pred:{cand_pred}")
                return True, cand_pred
                # 没预测成功的也记录，分析样本的一个小步骤用到了

            # 使用最佳候选更新代码
            if best_candidate:
                k, uid, cand_str, new_logit_val = best_candidate
                x_str = cand_str
                # 更新token列表和标识符集合
                x_raw = [uid if token == k else token for token in x_raw]
                identifiers.remove(k)
                identifiers.add(uid)
                base_true_logit = new_logit_val
                last_improvement = iter_idx
                iter_success = True
                logger.debug(
                    f"UPDATE: ID={index} | {k}=>{uid} | Logit:{new_logit_val:.3f} | Improvement:{best_improvement:.3f}")

                # 显著改进后减少最大迭代次数
                if best_improvement > 1.0 and iter_idx > 2:
                    n_iter = min(n_iter, last_improvement + 5)
                    logger.debug(f"Reduced max iterations to {n_iter}")

            # 提前终止条件：连续3轮无改进
            if not iter_success:
                if iter_idx - last_improvement > 3 and iter_idx > 4:
                    logger.info(f"Early termination at iteration {iter_idx + 1} | ID={index}")
                    break

        self.adv_record.write(index, x_str, None, true_label, base_pred, None)
        # 没攻击成功的也记录一下
        # 攻击失败
        logger.info(f"FAIL! ID={index} | Final logit:{base_true_logit:.3f}")
        return False, true_label

    def attack_all(self, n_candidate=100, n_iter=20):
        """批量攻击测试集"""

        n_succ = 0
        total_count = self.dataset.test.get_size()

        start_time = time.time()
        logger.info(f"Starting attack on {total_count} samples")

        # 进度条和统计
        progress_bar = tqdm(
            total=total_count,
            desc="Attacking",
            unit="sample",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        attack_stats = {
            'success': 0,
            'original_errors': 0,
            'total': 0,
            'total_time': 0.0
        }

        for i in range(total_count):
            batch = self.dataset.test.next_batch(1)
            sample_id = i
            true_label = batch['y'][0]
            attack_stats['total'] += 1

            # 获取当前样本的原始token列表和代码字符串
            x_raw = batch['raw'][0]  # token列表
            x_str = batch['original_code'][0]  # 原始代码字符串

            t_start = time.time()
            success, pred = self.attack(
                x_raw,
                x_str,
                true_label,
                sample_id,
                n_candidate,
                n_iter
            )
            elapsed = time.time() - t_start
            attack_stats['total_time'] += elapsed

            if success:
                n_succ += 1
                attack_stats['success'] += 1
                logger.info(f"Attack success! ID={sample_id} | Time: {elapsed:.2f}s")
            elif pred != true_label:
                attack_stats['original_errors'] += 1
                logger.info(f"Original error. ID={sample_id} | True:{true_label} Pred:{pred}")

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({
                "success_rate": f"{attack_stats['success'] / attack_stats['total']:.2%}",
                "avg_time": f"{attack_stats['total_time'] / attack_stats['total']:.2f}s"
            })

        total_time = time.time() - start_time
        avg_time_per_sample = total_time / total_count if total_count else 0
        success_rate = attack_stats['success'] / (total_count - attack_stats['original_errors']) \
            if total_count - attack_stats['original_errors'] > 0 else 0

        # 最终报告
        logger.info(f"\n{'=' * 50}")
        logger.info(f"[FINAL STATS]")
        logger.info(f"Total samples: {total_count}")
        logger.info(f"Original errors: {attack_stats['original_errors']}")
        logger.info(f"Successful attacks: {attack_stats['success']}")
        logger.info(f"Attack success rate: {success_rate:.2%}")
        logger.info(f"Total time: {total_time:.2f}s | Avg time per sample: {avg_time_per_sample:.2f}s")

        # 关闭记录器
        if self.adv_record:
            self.adv_record.close()
        if self.disturb_recoder:
            self.disturb_recoder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    # 数据路径参数
    parser.add_argument('--train_data', type=str, help="Path to training data JSONL file")
    parser.add_argument('--valid_data', type=str, help="Path to validation data JSONL file")
    parser.add_argument('--test_data', type=str, required=True, help="Path to test data JSONL file")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to trained model directory")
    parser.add_argument('--block_size', type=int, default=512)

    # 模型参数
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str)
    parser.add_argument("--config_name", default="microsoft/codebert-base", type=str)

    parser.add_argument("--adv_store_path", type=str, help="Path to store the adv CSV file")
    parser.add_argument("--disturb_store_path", type=str, help="Path to store the disturb CSV file")
    parser.add_argument('--seed', type=int, default=123456, help="random seed for initialization")
    parser.add_argument('--n_class', type=int, default=1, help="Number of classes")

    # 攻击参数
    parser.add_argument('--n_candidate', type=int, default=5, help="Number of candidates per token")
    parser.add_argument('--n_iter', type=int, default=40, help="Max attack iterations per sample")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    args = parser.parse_args()
    args.device = torch.device("cuda")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 66
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
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

    model = CodeBERTModel(model, config, tokenizer, args)

    checkpoint_prefix = 'best_acc_model/model.bin'
    output_dir = os.path.join(args.model_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir), strict=False)
    model.to(args.device)

    # 加载数据集
    vul_dataset = VulDataset(
        train_path=args.train_data,
        valid_path=args.valid_data,
        test_path=args.test_data,
        tokenizer=tokenizer
    )

    # 初始化记录器
    adv_record = Adversarial_Recorder(args.adv_store_path) if args.adv_store_path else None
    disturb_recoder = Disturb_Recorder(args.disturb_store_path) if args.disturb_store_path else None

    # 初始化攻击器并开始攻击
    atk = Attacker(vul_dataset, model, tokenizer, adv_record, disturb_recoder, args.device)
    atk.attack_all(n_candidate=args.n_candidate, n_iter=args.n_iter)
