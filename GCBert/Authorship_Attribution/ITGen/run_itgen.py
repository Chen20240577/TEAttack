import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_itgen.py \
        --output_dir=../saved_models \
        --model_type=roberta \
        --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
        --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
        --csv_store_path analysis/attack_itgen_all.jsonl \
        --base_model=../../../../../Models/microsoft/graphcodebert-base \
        --eval_data_file=../../../Datasets/Authorship_Attribution/graphcodebert-base/processed_gcjpy/valid.txt \
        --code_length 384 \
        --data_flow_length 128 \
        --eval_batch_size 2 \
        --seed 123456 2>&1| tee ../logs/ITGen_GCBert.log")
