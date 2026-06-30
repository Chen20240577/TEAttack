import os

# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/Alert_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Alert_Bert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_Alert_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/Alert_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Alert_GCBert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_Alert_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/Alert_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Alert_T5_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_Alert_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/Alert_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Alert_GPT_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_Alert_GPT.log")


os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/Carrot_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Carrot_Bert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_Carrot_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/Carrot_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Carrot_GCBert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_Carrot_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/Carrot_T5.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Carrot_T5_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_Carrot_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/Carrot_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/Carrot_GPT_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_Carrot_GPT.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/RNNS_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/RNNS_Bert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_RNNS_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/RNNS_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/RNNS_GCBert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_RNNS_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/RNNS_T5.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/RNNS_T5_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_RNNS_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/RNNS_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/RNNS_GPT_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_RNNS_GPT.log")
#
#
#
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/MHM_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/MHM_Bert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_MHM_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/MHM_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/MHM_GCBert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_MHM_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/MHM_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/MHM_T5_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_MHM_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/MHM_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/MHM_GPT_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_MHM_GPT.log")
#
#
#
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/WIR_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/WIR_Bert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_WIR_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/WIR_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/WIR_GCBert_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_WIR_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/WIR_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/WIR_T5_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_WIR_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --config_name=../../../../../Models/microsoft/codebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/codebert-base \
#     --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
#     --csv_read_path ../../../AdvExamples/Clone_detection/WIR_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/WIR_GPT_transfer.csv \
#     --eval_batch_size 32 \
#     --block_size 512 \
#     --seed 123456 2>&1| tee ../logs/transfer_WIR_GPT.log")
#
#
#

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/ACCENT_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/ACCENT_Bert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_ACCENT_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/ACCENT_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/ACCENT_GCBert_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_ACCENT_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/ACCENT_T5.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/ACCENT_T5_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_ACCENT_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --config_name=../../../../../Models/microsoft/codebert-base \
    --model_name_or_path=../../../../../Models/microsoft/codebert-base \
    --tokenizer_name=../../../../../Models/FacebookAI/roberta-base \
    --csv_read_path ../../../AdvExamples/Clone_detection/ACCENT_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Clone_detection/CodeBert/ACCENT_GPT_transfer.csv \
    --eval_batch_size 32 \
    --block_size 512 \
    --seed 123456 2>&1| tee ../logs/transfer_ACCENT_GPT.log")
