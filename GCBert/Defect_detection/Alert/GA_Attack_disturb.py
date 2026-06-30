# -*- coding: utf-8 -*-
# python GA-Attack.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --adv_store_path ../../../AdvExamples/Defect_detection/Alert_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --use_ga \
    --eval_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/test_subs_0_400.jsonl \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/Alert_GCBert.log")
