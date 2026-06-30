# -*- coding: utf-8 -*-
# python substitutes.py

import os

os.system("python ./get_substitutes.py \
    --store_path ./dataset/test_subs_0_400.jsonl \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --eval_data_file=./dataset/test.jsonl \
    --block_size 512 \
    --index 0 400")
