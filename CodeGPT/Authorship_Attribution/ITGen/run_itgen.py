import os

os.system("CUDA_VISIBLE_DEVICES=0 python attack_itgen.py \
        --output_dir=../saved_models \
        --model_type=gpt2 \
        --tokenizer_name=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
        --model_name_or_path=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
        --csv_store_path analysis/attack_itgen_all.jsonl \
        --base_model=../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2 \
        --eval_data_file=../../../Datasets/Authorship_Attribution/codegpt-small/processed_gcjpy/valid.txt \
        --block_size 512 \
        --eval_batch_size 2 \
        --seed 123456 2>&1| tee ../logs/ITGen_GPT.log")
