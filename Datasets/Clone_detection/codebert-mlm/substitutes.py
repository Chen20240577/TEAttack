# -*- coding: utf-8 -*-
# python substitutes.py

import os

# os.system("CUDA_VISIBLE_DEVICES=0 python ./get_substitutes.py\
#     --store_path ./data_folder/test_subs_0_10.jsonl\
#     --base_model=../../../../../Models/microsoft/codebert-base-mlm\
#     --eval_data_file=./data_folder/test_sampled.txt\
#     --block_size 512\
#     --index 0 10")

os.system("CUDA_VISIBLE_DEVICES=0 python ./get_substitutes.py\
    --store_path ./data_folder/test_subs_0_500.jsonl\
    --base_model=../../../../../Models/microsoft/codebert-base-mlm\
    --eval_data_file=./data_folder/test_sampled.txt\
    --block_size 512\
    --index 0 500")
