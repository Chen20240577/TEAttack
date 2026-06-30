# -*- coding: utf-8 -*-
# GA_Attack_disturb.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --adv_store_path ../../../AdvExamples/Defect_detection/Alert_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --base_model=../../../../../Models/microsoft/codebert-base-mlm \
    --use_ga \
    --eval_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/Alert_T5.log")

# os.system("CUDA_VISIBLE_DEVICES=0 python ./record_attack.py \
#     --output_dir=../saved_models \
#     --model_type=t5 \
#     --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
#     --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
#     --adv_store_path ../../../AdvExamples/Defect_detection/Alert_T5_test.csv \
#     --disturb_store_path ./analysis/disturbed_test.csv \
#     --base_model=../../../../../Models/microsoft/codebert-base-mlm \
#     --use_ga \
#     --eval_data_file=../../../Datasets/Defect_detection/codet5-base/dataset/test_subs_0_10.jsonl \
#     --block_size 512 \
#     --eval_batch_size 32 \
#     --seed 123456  2>&1 | tee ../logs/Alert_T5_test.log")
