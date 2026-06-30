# -*- coding: utf-8 -*-
# python GA-Attack_disturb.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Alert_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --use_ga \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --train_data_file=../../../Datasets/Clone_detection/codet5-base/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codet5-base/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/codet5-base/data_folder/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee ../logs/Alert_T5.log")
