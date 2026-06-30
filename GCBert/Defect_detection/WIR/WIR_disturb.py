# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --number_labels 1 \
    --adv_store_path ../../../AdvExamples/Defect_detection/WIR_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type c \
    --eval_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/test_subs_0_400.jsonl \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/WIR_GCBert.log")
