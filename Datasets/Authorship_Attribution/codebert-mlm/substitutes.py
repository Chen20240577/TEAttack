# -*- coding: utf-8 -*-
# python substitutes.py

import os

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
