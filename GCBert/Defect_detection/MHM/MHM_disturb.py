# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --number_labels 1 \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --original \
    --adv_store_path ../../../AdvExamples/Defect_detection/MHM_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/train.jsonl \
    --eval_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/test_subs_0_400.jsonl \
    --test_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/test.jsonl \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/MHM_GCBert.log")
