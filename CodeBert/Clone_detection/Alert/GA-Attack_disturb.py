# -*- coding: utf-8 -*-
# python GA-Attack.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --adv_store_path ../../../AdvExamples/Clone_detection/Alert_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --use_ga \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --train_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/Alert_Bert.log")
# code_length要和微调的code_length一致
