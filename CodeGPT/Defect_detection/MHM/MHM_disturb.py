# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 1 \
    --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --original \
    --adv_store_path ../../../AdvExamples/Defect_detection/MHM_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Defect_detection/codegpt-small/dataset/train.jsonl \
    --eval_data_file=../../../Datasets/Defect_detection/codegpt-small/dataset/test_subs_0_400.jsonl \
    --test_data_file=../../../Datasets/Defect_detection/codegpt-small/dataset/test.jsonl \
    --block_size 512 \
    --eval_batch_size 16 \
    --seed 123456 2>&1| tee ../logs/MHM_GPT.log")
