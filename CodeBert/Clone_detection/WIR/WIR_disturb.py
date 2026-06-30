# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --number_labels 2 \
    --adv_store_path ../../../AdvExamples/Clone_detection/WIR_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type java \
    --eval_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled_0_500.txt \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee ../logs/WIR_Bert.log")
