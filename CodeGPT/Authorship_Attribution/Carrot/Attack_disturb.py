# -*- coding: utf-8 -*-
# python Attack_disturb.py

import os

os.system("python ./attacker.py \
    --model_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --adv_store_path ../../../AdvExamples/Authorship_Attribution/Carrot_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --test_data=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.jsonl \
    --seed 123456  2>&1 | tee ../logs/Carrot_GPT.log")
