# -*- coding: utf-8 -*-
# python substitutes.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./get_substitutes.py \
    --store_path ./processed_gcjpy/valid_subs.jsonl \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --eval_data_file=./processed_gcjpy/valid.txt \
    --block_size 512")
