# -*- coding: utf-8 -*-
# python GA-Attack_disturb.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Alert_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --use_ga \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --train_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled.txt \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/Alert_GCBert.log")
