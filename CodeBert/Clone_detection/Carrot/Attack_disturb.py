# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base\
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Carrot_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_subs_0_500.jsonl \
    --seed 123456  2>&1 | tee ../logs/Carrot_Bert.log")
