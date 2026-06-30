# -*- coding: utf-8 -*-
# python GA-Attack.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --number_labels 66 \
    --do_eval \
    --use_ga \
    --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Alert_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type python \
    --train_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/train.txt \
    --eval_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
    --test_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ../logs/Alert_GPT.log")
