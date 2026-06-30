# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --adv_store_path ../../../AdvExamples/Defect_detection/Carrot_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Defect_detection/codebert-mlm/dataset/test_subs_0_400.jsonl \
    --seed 123456  2>&1 | tee ../logs/Carrot_T5.log")
# --test_data=../preprocess/dataset/test_subs_0_10.jsonl
# test_subs_0_10是codebert生成的，cv过来的，但是不需要里面对应的替换词，所以直接拿来当测试集
# 0_400
