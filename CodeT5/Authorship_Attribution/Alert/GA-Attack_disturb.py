# -*- coding: utf-8 -*-
# python GA-Attack_disturb.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --number_labels 66 \
    --do_eval \
    --use_ga \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Alert_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type python \
    --train_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codet5-base/processed_gcjpy/valid.txt \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/Alert_T5.log")
#     --train_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt \
#     --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
#     --test_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
# 掩码语言模型用的codebert-mlm，理论上讲可以直接用对应的替换词数据集，但是出了点问题，缓存文件冲突，每次切换模型都要删除缓存文件，所以再cv个数据集
