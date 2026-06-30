# -*- coding: utf-8 -*-
# python ACCENT_disturb.py
# wrapper: 调用 accent_attack.py（类似 WIR_disturb.py 的用法）

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./accent_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --number_labels 66 \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/ACCENT_Bert.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --nearest_k_path ../../../W2V_Models/Authorship_Attribution/datasets/var_name/code_nearest_top30.pkl \
    --language_type python \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1 | tee ../logs/ACCENT_Bert.log")
