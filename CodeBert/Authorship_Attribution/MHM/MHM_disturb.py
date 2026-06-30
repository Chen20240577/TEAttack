# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --number_labels 66 \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --original \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/MHM_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456 2>&1| tee ../logs/MHM_Bert.log")
