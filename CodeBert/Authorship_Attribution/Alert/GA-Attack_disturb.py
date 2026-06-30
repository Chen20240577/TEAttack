# -*- coding: utf-8 -*-
# python GA-Attack.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --number_labels 66 \
    --do_eval \
    --use_ga \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Alert_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type python \
    --train_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/Alert_Bert.log")

# # 更换掩码模型
# os.system("python ./record_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --number_labels 66 \
#     --do_eval \
#     --use_ga \
#     --base_model=../../../../../Models/microsoft/codebert-base \
#     --adv_store_path ../../../AdvExamples/Authorship_Attribution/Alert_Bert_base.csv \
#     --disturb_store_path ./analysis/disturbed.csv \
#     --language_type python \
#     --train_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt \
#     --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/codebert-base/valid.txt \
#     --test_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/codebert-base/valid.txt \
#     --block_size 512 \
#     --train_batch_size 8 \
#     --eval_batch_size 32 \
#     --evaluate_during_training \
#     --seed 123456 2>&1| tee ../logs/Alert_Bert.log")
