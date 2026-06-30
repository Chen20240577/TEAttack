# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --number_labels 2 \
    --adv_store_path ../../../AdvExamples/Clone_detection/WIR_GCBert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type java \
    --eval_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled_0_500.txt \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee ../logs/WIR_GCBert.log")
