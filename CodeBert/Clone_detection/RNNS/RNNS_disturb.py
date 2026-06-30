# -*- coding: utf-8 -*-
# python RNNS_disturb.py

import os

os.system("python rnns_record_attacker.py \
    --tgt_model=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --rnns_type=RNNS-Smooth \
    --max_distance=0.15 \
    --max_length_diff=3 \
    --substitutes_size=30  \
    --iters=6 \
    --a=0.2 \
    --number_labels 2 \
    --adv_store_path ../../../AdvExamples/Clone_detection/RNNS_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --all_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/data.jsonl \
    --eval_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled_0_500.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/RNNS_Bert.log")
# all_data_file 用来传入整个数据集，会需要用到整个数据集的参数集
