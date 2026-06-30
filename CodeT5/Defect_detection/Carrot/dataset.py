# -*- coding: utf-8 -*-
"""
@author: einsam
"""

import copy
import json
import random
import re
from typing import List, Optional

import numpy


class Dataset(object):

    def __init__(self, ys: List[int] = [],
                 raws: Optional[List[List[str]]] = None,
                 dtype=None, original_codes=None):

        self.__dtype = dtype
        self.__raws = []
        self.__ys = []
        self.__original_codes = original_codes if original_codes is not None else {}

        if raws is None:
            raws = [(None) for _ in ys]
        else:
            assert len(ys) and len(ys) == len(raws)

        self.__size = len(ys)

        for i, (y, r) in enumerate(zip(ys, raws)):
            self.__raws.append(r)
            self.__ys.append(y)

        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])

        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        self.__epoch = random.sample(range(self.__size), self.__size)

    def next_batch(self, batch_size=32):
        batch = {"y": [], "raw": [], "new_epoch": False, "original_code": []}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
            # 确保有足够样本
            if len(self.__epoch) < batch_size:
                raise ValueError(f"Not enough samples ({len(self.__epoch)}) for batch size {batch_size}")

        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]

        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['raw'] = copy.deepcopy([self.__raws[i] for i in idxs])

        # 添加原始代码字符串
        batch['original_code'] = [self.__original_codes.get(i, "") for i in idxs]
        return batch

    def get_size(self):
        return self.__size

    def get_rest_epoch_size(self):
        return len(self.__epoch)

    def get_original_code(self, index):
        return self.__original_codes.get(index, "")


class VulDataset(object):
    """VulDataset类，支持多个JSONL文件"""

    def __init__(self, train_path=None, valid_path=None, test_path=None,
                 dtype='32', tokenizer=None):

        self.__dtypes = self.__dtype(dtype)
        self.tokenizer = tokenizer
        self.__txt2idx = tokenizer.get_vocab()

        # 初始化original_codes字典
        self.original_codes = {}

        # 加载训练集
        self.train = self.__load_dataset(train_path) if train_path else None

        # 加载验证集
        self.dev = self.__load_dataset(valid_path) if valid_path else None

        # 加载测试集
        self.test = self.__load_dataset(test_path) if test_path else None

    def __load_dataset(self, path):
        """从JSONL文件加载数据集"""
        if not path:
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
        except Exception as e:
            print(f"Error loading dataset from {path}: {e}")
            return None

        ys = []
        raws = []

        count = 0
        for i, d in enumerate(data):
            try:
                code = d.get('func', '')
                label = d.get('target', 0)

                # 存储原始代码字符串（去除冗余空格）
                clean_code = re.sub(r'\s+', ' ', code).strip()
                # 添加到全局字典
                self.original_codes[count] = clean_code
                count += 1

                # 分词处理
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|\S', code)
                raws.append(tokens)

                ys.append(label)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        return Dataset(
            ys=ys, raws=raws,
            dtype=self.__dtypes,
            original_codes=self.original_codes
        )

    def get_original_code(self, index):
        """根据索引获取原始代码字符串"""
        return self.original_codes.get(index, "")

    def __dtype(self, dtype='32'):
        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):
        return self.__dtypes

    def get_txt2idx(self):
        return copy.deepcopy(self.__txt2idx)

    def vocab2idx(self, vocab):
        if vocab in self.__txt2idx.keys():
            return self.__txt2idx[vocab]
        else:
            return self.__txt2idx['<unk>']
