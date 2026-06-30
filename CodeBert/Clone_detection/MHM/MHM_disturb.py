# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --number_labels 2 \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --original \
    --adv_store_path ../../../AdvExamples/Clone_detection/MHM_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/MHM_Bert.log")
