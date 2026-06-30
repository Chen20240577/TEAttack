# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Carrot_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Clone_detection/codet5-base/data_folder/test_subs_0_500.jsonl \
    --seed 123456  2>&1 | tee ../logs/Carrot_T5.log")
