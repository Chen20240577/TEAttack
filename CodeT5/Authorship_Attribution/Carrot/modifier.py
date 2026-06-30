# -*- coding: utf-8 -*-
"""
@author: einsam
"""

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# 配置日志
logger = logging.getLogger('TokenModifier')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class TokenModifier:
    """
    用于修改代码中的标识符(token)，生成对抗样本

    参数:
        classifier: 目标分类器模型
        loss: 损失函数
        uids: 所有可能的标识符列表
        txt2idx: 文本到索引的映射字典
        idx2txt: 索引到文本的映射字典
        tokenizer: 用于分词和编码的tokenizer
        max_workers: 最大并行进程数
    """
    """
        优化的标识符修改器，用于生成对抗样本

        主要优化:
        1. 多线程处理候选生成
        2. 缓存机制减少重复计算
        3. 更高效的token处理
        4. 梯度计算的批量化
    """

    def __init__(self,
                 classifier: nn.Module,
                 uids: List[str],
                 tokenizer,
                 device,
                 max_workers: int = 4):
        # def __init__(self,
        #              classifier: nn.Module,
        #              loss: nn.Module,
        #              uids: List[str],
        #              txt2idx: Dict[str, int],
        #              idx2txt: Dict[int, str],
        #              tokenizer,
        #              device,
        #              max_workers: int = 4):

        self.tokenizer = tokenizer
        self.classifier = classifier
        # self.loss = loss
        # self.txt2idx = txt2idx
        # self.idx2txt = idx2txt
        self.device = device
        self.vocab_size = len(tokenizer)

        # 设置并行工作线程数
        self.max_workers = max_workers
        logger.info(f"Initializing TokenModifier with {self.max_workers} workers")

        # 关键字过滤
        forbidden_uid = {"auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
                         "enum", "extern", "float", "for", "goto", "if", "int", "long", "register", "restrict",
                         "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
                         "unsigned", "void", "volatile", "while", "printf", "scanf", "malloc", "free"}
        logger.debug(f"Forbidden keywords: {sorted(forbidden_uid)}")

        # 缓存候选标识符
        self._uids = []
        self.__candidate_cache = {}  # 缓存候选替换结果
        self._candidate_gen_cache = {}  # 缓存候选生成结果

        # 动态构建候选UID
        start_time = time.time()
        valid_tokens = 0
        for token in uids:
            if token in forbidden_uid:
                continue
            try:
                # 检查token是否在tokenizer词表中
                encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1:  # 确保是单一token
                    self._uids.append(encoded[0])
                    valid_tokens += 1
            except:
                continue

        logger.info(
            f"Loaded {valid_tokens} valid identifiers from {len(uids)} tokens in {time.time() - start_time:.2f}s")

        # 创建UID掩码
        self.uids = self.__gen_uid_mask_on_vocab(self._uids)
        logger.debug(f"UID mask size: {len(self._uids)} | Mask shape: {self.uids.shape}")

    def __tokens_to_code(self, tokens: List[str]) -> str:
        """保留原始格式的重建"""
        code = ""
        for token in tokens:
            if token.startswith('Ġ'):  # RoBERTa风格空格
                code += ' ' + token[1:]
            elif token in {',', ';', '}', ']', ')', '.'}:
                code += token
            else:
                code += ' ' + token
        return code.strip()

    def _is_punctuation(self, token):
        """检查token是否是标点符号"""
        return token in {',', ';', '.', '(', ')', '{', '}', '[', ']', '=', '+', '-', '*', '/'}

    def __gen_uid_mask_on_vocab(self, uids: List[int]) -> torch.Tensor:
        """生成标识符掩码，标记哪些token是有效的标识符"""
        if not uids:
            return torch.zeros(self.vocab_size, 1, device=self.device)

        indices = torch.LongTensor(uids).to(self.device)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        mask[indices] = True
        return mask.reshape([self.vocab_size, 1])

    def __process_single_candidate(
            self, x_raw: List[str], ori_uid_raw: str, new_uid_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """处理单个候选标识符替换"""
        try:
            # 解码新标识符
            new_uid_raw = self.tokenizer.decode([new_uid_idx])
            if not new_uid_raw or new_uid_raw == ori_uid_raw:
                return None, None

            # 缓存检查
            cache_key = (tuple(x_raw), ori_uid_raw, new_uid_idx)
            if cache_key in self._candidate_gen_cache:
                return self._candidate_gen_cache[cache_key]

            # 应用替换
            new_x_tokens = [new_uid_raw if t == ori_uid_raw else t for t in x_raw]

            # 转换回代码字符串
            modified_code = self.__tokens_to_code(new_x_tokens)

            # 缓存结果
            self._candidate_gen_cache[cache_key] = (modified_code, new_uid_raw)
            return modified_code, new_uid_raw
        except Exception as e:
            logger.debug(f"Error generating candidate: {str(e)}")
            return None, None

    def rename_uid(self,
                   x_raw: List[str],
                   y: int,
                   ori_uid_raw: str,
                   n_candidate: int = 5) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        优化的重命名方法 - 减少计算和批量处理

        返回:
            new_x_strs: 修改后的代码字符串列表
            new_x_uid: 新标识符列表
        """
        start_time = time.time()
        cache_key = (tuple(x_raw), y, ori_uid_raw, n_candidate)

        # 检查缓存
        if cache_key in self.__candidate_cache:
            logger.debug(f"Cache hit for {ori_uid_raw}")
            return self.__candidate_cache[cache_key]

        # 处理代码中的标识符
        try:
            # 更高效的token检查
            encoded = self.tokenizer(ori_uid_raw, add_special_tokens=False)["input_ids"]
            if not encoded or len(encoded) != 1:
                logger.debug(f"Skipping {ori_uid_raw}: not a single token")
                return None, None

            Gori_uid_idx = encoded[0]
            if Gori_uid_idx >= len(self.uids) or not self.uids[Gori_uid_idx]:
                logger.debug(f"Skipping {ori_uid_raw}: not in valid identifiers")
                return None, None
        except Exception as e:
            logger.error(f"Error encoding token {ori_uid_raw}: {str(e)}")
            return None, None

        # 并行生成候选标识符
        candidate_args = [(x_raw, ori_uid_raw, new_uid_idx) for new_uid_idx in self._uids]
        new_x_strs = []
        new_x_uid = []

        # 使用线程池并行处理
        if self.max_workers > 1:
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    # 限制生成的候选数量
                    for args in candidate_args[:min(1000, len(candidate_args))]:
                        futures.append(executor.submit(self.__process_single_candidate, *args))

                    for future in futures:
                        result = future.result()
                        if result and result[0]:
                            new_x_strs.append(result[0])
                            new_x_uid.append(result[1])
                            if len(new_x_strs) >= n_candidate * 2:  # 生成额外候选供选择
                                break
            except Exception as e:
                logger.error(f"Thread pool error: {str(e)}")

        # 如果并行失败或禁用了并行
        if not new_x_strs:
            for i, args in enumerate(candidate_args):
                if len(new_x_strs) >= n_candidate * 2:
                    break
                result = self.__process_single_candidate(*args)
                if result and result[0]:
                    new_x_strs.append(result[0])
                    new_x_uid.append(result[1])

        # 如果生成了太多候选，随机选择部分
        if len(new_x_strs) > n_candidate * 1.5:
            indices = np.random.choice(len(new_x_strs), size=min(len(new_x_strs), n_candidate), replace=False)
            new_x_strs = [new_x_strs[i] for i in indices]
            new_x_uid = [new_x_uid[i] for i in indices]

        # 缓存结果
        if new_x_strs:
            self.__candidate_cache[cache_key] = (new_x_strs, new_x_uid)

        logger.debug(f"Generated {len(new_x_strs)} candidates for {ori_uid_raw} in {time.time() - start_time:.3f}s")
        return new_x_strs, new_x_uid

    def _get_batched_data(self, raws: List[List[str]], ys: List[int]) -> Dict[str, Any]:
        """创建批次数据，用于模型输入"""
        # 直接处理token列表，转换为字符串
        xs = [self.__tokens_to_code(raw) for raw in raws]

        batch = {
            "x": xs,
            "y": ys,
            "raw": copy.deepcopy(raws),
            "id": None,
            "new_epoch": False
        }
        return batch
