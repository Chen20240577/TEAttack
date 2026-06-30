# -*- coding: utf-8 -*-
# python WIR_disturb.py
import os

os.system("CUDA_VISIBLE_DEVICES=0 python ./wir_attack.py \
    --output_dir=../saved_models \
    --model_type=t5 \
    --config_name=../../../../../Models/Salesforce/codet5-base \
    --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
    --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
    --number_labels 2 \
    --adv_store_path ../../../AdvExamples/Clone_detection/WIR_T5.csv \
    --disturb_store_path ./analysis/disturbed.csv \
    --language_type java \
    --eval_data_file=../../../Datasets/Clone_detection/codet5-base/data_folder/test_sampled_0_500.txt \
    --block_size 512 \
    --eval_batch_size 4 \
    --seed 123456 2>&1| tee ../logs/WIR_T5.log")
