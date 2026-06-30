# -*- coding: utf-8 -*-
# python transfer.py

import os

# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/Alert_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Alert_Bert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_Alert_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/Alert_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Alert_GCBert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_Alert_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/Alert_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Alert_T5_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_Alert_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/Alert_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Alert_GPT_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_Alert_GPT.log")
#
#
#

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/Carrot_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Carrot_Bert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_Carrot_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/Carrot_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Carrot_GCBert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_Carrot_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/Carrot_T5.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Carrot_T5_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_Carrot_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/Carrot_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/Carrot_GPT_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_Carrot_GPT.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_Bert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_RNNS_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_GCBert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_RNNS_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_T5.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_T5_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_RNNS_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_GPT_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_RNNS_GPT.log")

#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/MHM_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/MHM_Bert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_MHM_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/MHM_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/MHM_GCBert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_MHM_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/MHM_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/MHM_T5_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_MHM_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/MHM_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/MHM_GPT_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_MHM_GPT.log")
#
#
#
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/WIR_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/WIR_Bert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_WIR_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/WIR_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/WIR_GCBert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_WIR_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/WIR_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/WIR_T5_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_WIR_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/WIR_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/WIR_GPT_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_WIR_GPT.log")
#
#
#
#
os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/ACCENT_Bert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/ACCENT_Bert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_ACCENT_Bert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/ACCENT_GCBert.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/ACCENT_GCBert_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_ACCENT_GCBert.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/ACCENT_T5.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/ACCENT_T5_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_ACCENT_T5.log")

os.system("python csv_attack.py \
    --output_dir=../saved_models \
    --model_type=roberta \
    --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
    --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
    --csv_read_path ../../../AdvExamples/Defect_detection/ACCENT_GPT.csv \
    --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/ACCENT_GPT_transfer.csv \
    --code_length 384 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee ../logs/transfer_ACCENT_GPT.log")

# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_limit_Bert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_limit_Bert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_RNNS_limit_Bert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_limit_GCBert.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_limit_GCBert_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_RNNS_limit_GCBert.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_limit_T5.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_limit_T5_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_RNNS_limit_T5.log")
#
# os.system("python csv_attack.py \
#     --output_dir=../saved_models \
#     --model_type=roberta \
#     --tokenizer_name=../../../../../Models/microsoft/graphcodebert-base \
#     --model_name_or_path=../../../../../Models/microsoft/graphcodebert-base \
#     --csv_read_path ../../../AdvExamples/Defect_detection/RNNS_limit_GPT.csv \
#     --transfer_store_path ../../../TransfAEs/Defect_detection/GCBert/RNNS_limit_GPT_transfer.csv \
#     --code_length 384 \
#     --data_flow_length 128 \
#     --eval_batch_size 8 \
#     --seed 123456  2>&1 | tee ../logs/transfer_RNNS_limit_GPT.log")
