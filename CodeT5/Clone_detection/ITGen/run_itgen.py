import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_itgen.py \
        --output_dir=../saved_models \
        --model_type=t5 \
        --tokenizer_name=../../../../../Models/Salesforce/codet5-base \
        --model_name_or_path=../../../../../Models/Salesforce/codet5-base \
        --csv_store_path analysis/attack_itgen_all.jsonl \
        --base_model=../../../../../Models/microsoft/codebert-base-mlm \
        --eval_data_file=../../../Datasets/Clone_detection/codet5-base/data_folder/test_sampled_0_500.txt \
        --block_size 512 \
        --eval_batch_size 2 \
        --seed 123456 2>&1| tee ../logs/ITGen_T5.log")
