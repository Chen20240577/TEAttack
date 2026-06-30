# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base\
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Carrot_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_subs_0_500.jsonl \
    --code_length 384 \
    --data_flow_length 128 \
    --seed 123456  2>&1 | tee ../logs/Carrot_GCBert.log")
