# -*- coding: utf-8 -*-
# python MHM_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./mhm_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 2 \
    --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --original \
    --adv_store_path ../../../AdvExamples/Clone_detection/MHM_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --train_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 8 \
    --seed 123456 2>&1| tee ../logs/MHM_GPT.log")
