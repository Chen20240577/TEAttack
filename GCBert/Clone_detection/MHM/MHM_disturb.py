# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --number_labels 2 \
    --base_model=../../../../../Models/microsoft/graphcodebert-base \
    --original \
    --adv_store_path ../../../AdvExamples/Clone_detection/MHM_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled.txt \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/MHM_GCBert.log")
