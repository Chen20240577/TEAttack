# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --number_labels 1 \
    --adv_store_path ../../../AdvExamples/Defect_detection/WIR_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type c \
    --eval_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee ../logs/WIR_T5.log")
