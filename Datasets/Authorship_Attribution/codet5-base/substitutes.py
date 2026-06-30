# -*- coding: utf-8 -*-
# python substitutes.py

import os

# codet5-base掩码语言模型用的codebert-mlm，理论上讲可以直接用对应的替换词数据集，但是出了点问题，缓存文件冲突，每次切换模型都要删除缓存文件，所以再cv个数据集

os.system("CUDA_VISIBLE_DEVICES=0 python ./get_substitutes.py \
    --store_path ./processed_gcjpy/valid_subs.jsonl \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --eval_data_file=./processed_gcjpy/valid.txt \
    --block_size 512")

# 用codebert生成替换词
# os.system("CUDA_VISIBLE_DEVICES=0 python ./get_substitutes.py \
#     --store_path ./processed_gcjpy/codebert-base/valid_subs.jsonl \
#     --base_model=../../../../../Models/microsoft/codebert-base \
#     --eval_data_file=./processed_gcjpy/codebert-base/valid.txt \
#     --block_size 512")
