# -*- coding: utf-8 -*-
# python GA-Attack_disturb.py

import os

os.system("python ./record_attack.py \
    --output_dir=../saved_models \
    --model_type=gpt2 \
    --config_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --adv_store_path ../../../AdvExamples/Clone_detection/Alert_GPT.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --use_ga \
    --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
    --train_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/train_sampled.txt \
    --eval_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/test_sampled_0_500.txt \
    --test_data_file=../../../Datasets/Clone_detection/codegpt-small/data_folder/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee ../logs/Alert_GPT.log")
# code_length要和微调的code_length一致
