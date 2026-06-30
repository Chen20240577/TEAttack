# -*- coding: utf-8 -*-
# python RNNS_disturb.py

import os

os.system("python rnns_record_attacker.py \
    --tgt_model=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --rnns_type=RNNS-Smooth \
    --max_distance=0.15 \
    --max_length_diff=3 \
    --substitutes_size=30  \
    --iters=6 \
    --a=0.2 \
    --number_labels 1 \
    --adv_store_path ../../../AdvExamples/Defect_detection/RNNS_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/train.jsonl \
    --valid_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/valid.jsonl \
    --test_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/test.jsonl \
    --eval_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/RNNS_T5.log")
# --all_data_file=../../../Datasets/Defect_detection/codebert-mlm/function.jsonl \
# function直接处理有点麻烦，虽然作为整体数据集是合适的，但是用预处理后的 train.jsonl, valid.jsonl, test.jsonl 更方便
