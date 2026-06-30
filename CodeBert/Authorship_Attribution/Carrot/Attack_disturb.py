# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base\
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Carrot_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.jsonl \
    --seed 123456  2>&1 | tee ../logs/Carrot_Bert.log")
# --test_data=test_subs_0_10.jsonl
# test_subs_0_10是codebert生成的，cv过来的，但是不需要里面对应的替换词，所以直接拿来当测试集
# 0_400
