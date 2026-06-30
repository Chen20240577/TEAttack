# -*- coding: utf-8 -*-
# python fine-tuning.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./run.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 66 \
    --do_train \
    --train_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
    --epoch 30 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee ../logs/train.log")
