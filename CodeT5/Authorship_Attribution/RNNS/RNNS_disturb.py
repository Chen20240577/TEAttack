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
    --number_labels 66 \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/RNNS_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee ../logs/RNNS_T5.log")

#     --train_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt \
#     --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
#     --test_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
# 掩码语言模型用的codebert-mlm，理论上讲可以直接用对应的替换词数据集，但是出了点问题，缓存文件冲突，每次切换模型都要删除缓存文件，所以再cv个数据集
