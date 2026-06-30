# -*- coding: utf-8 -*-
# python GA_Attack_disturb.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --number_labels 66 \
    --do_eval \
    --use_ga \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Alert_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/valid.txt \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/Alert_GCBert.log")
