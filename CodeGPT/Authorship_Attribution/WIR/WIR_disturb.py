# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 66 \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/WIR_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type python \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/WIR_GPT.log")
