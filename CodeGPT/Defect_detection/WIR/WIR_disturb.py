# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 1 \
    --adv_store_path ../../../AdvExamples/Defect_detection/WIR_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type c \
    --eval_data_file=../../../Datasets/Defect_detection/codegpt-small/dataset/test_subs_0_400.jsonl \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/WIR_GPT.log")
