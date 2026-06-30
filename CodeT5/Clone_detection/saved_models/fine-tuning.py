# -*- coding: utf-8 -*-
# python fine-tuning.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./run.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --do_train \
    --train_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/valid_sampled.txt \
    --test_data_file=../../../Datasets/Clone_detection/codebert-mlm/data_folder/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/train.log")
