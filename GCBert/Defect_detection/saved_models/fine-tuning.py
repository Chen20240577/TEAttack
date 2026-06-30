# -*- coding: utf-8 -*-
# python fine-tuning.py

import os

os.system("python ./run.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/train.jsonl \
    --eval_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/valid.jsonl \
    --test_data_file=../../../Datasets/Defect_detection/graphcodebert-base/dataset/test.jsonl \
    --epoch 5 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train.log")
