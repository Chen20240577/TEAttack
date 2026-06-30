# -*- coding: utf-8 -*-
# python fine-tuning.py

import os

os.system("python ./run.py \
    --output_dir=../saved_models \
    --config_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --do_train \
    --language_type python \
    --number_labels 66 \
    --train_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/valid.txt \
    --epoch 30 \
    --code_length 384 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log")
