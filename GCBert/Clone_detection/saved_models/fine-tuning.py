# -*- coding: utf-8 -*-
# python fine-tuning.py

import os

os.system("python ./run.py \
    --output_dir=../saved_models \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/valid_sampled.txt \
    --test_data_file=../../../Datasets/Clone_detection/graphcodebert-base/data_folder/test_sampled.txt \
    --epoch 2 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 6 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log")
