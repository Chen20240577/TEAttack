# -*- coding: utf-8 -*-
# GA_Attack.py

import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --adv_store_path ../../../AdvExamples/Defect_detection/Alert_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --use_ga \
    --eval_data_file=../../../Datasets/Defect_detection/codegpt-small/dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee ../logs/Alert_GPT.log")
