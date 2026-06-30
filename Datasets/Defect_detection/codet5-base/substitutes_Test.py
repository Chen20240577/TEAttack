# -*- coding: utf-8 -*-
# python substitutes_Test.py

import os

os.system("python ./get_substitutes.py \
    --store_path ./dataset/test_subs_0_400.jsonl \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 0 400")
# codet5-base掩码语言模型用的codebert-mlm，理论上讲可以直接用对应的替换词数据集，但是出了点问题，缓存文件冲突，每次切换模型都要删除缓存文件，所以再cv个数据集

# 生成一个10example的测试用的小数据集
# os.system("python ./get_substitutes.py \
#     --store_path ./dataset/code-bert-mlm/test_subs_0_10.jsonl \
#     --base_model=../../../../../Models/microsoft/codebert-base-mlm \
#     --eval_data_file=./dataset/test.jsonl \
#     --block_size 512 \
#     --index 0 10")
