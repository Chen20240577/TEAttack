#!/bin/bash
# 设置库路径
export LD_LIBRARY_PATH="/root/miniconda3/envs/CLH/lib:$LD_LIBRARY_PATH"

#nohup ./run_scripts.sh > ./run_scripts.log 2>&1 &
#tail -f ./run_scripts.log
#chmod +x run_scripts.sh

## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/saved_models
#python fine-tuning.py
#sleep 10
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/saved_models
#python fine-tuning.py
#sleep 10
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/saved_models
#python fine-tuning.py
#
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

#cd /root/autodl-tmp/Experiments/Transferability/Datasets/Authorship_Attribution/codegpt-small
#python substitutes.py
#cd /root/autodl-tmp/Experiments/Transferability/Datasets/Clone_detection/codegpt-small
#python substitutes.py
#cd /Se-chenlinhua/Experiments/Transferability/Datasets/Defect_detection/graphcodebert-base
#python substitutes.py
#
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

# feature_adv_get和feature_disturb_get没必要跑，***_disturb.py其中的扰动也没有记录的必要，可以删掉
# 最初想看扰动的影响，但是研究核心是迁移性，方法内扰动过程和迁移性关系不大
# 目前准备通过迁移成功的AEs和没有迁移成功的AEs对比来得到结论

#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Authorship_Attribution/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Clone_detection/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Defect_detection/Carrot
#python Attack_disturb.py


#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Authorship_Attribution/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Clone_detection/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Defect_detection/Carrot
#python Attack_disturb.py
#
#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Authorship_Attribution/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Clone_detection/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Defect_detection/Carrot
#python Attack_disturb.py
#
#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Authorship_Attribution/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Clone_detection/Carrot
#python Attack_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Defect_detection/Carrot
#python Attack_disturb.py




#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Authorship_Attribution/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Clone_detection/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Defect_detection/RNNS
#python RNNS_disturb.py
#
#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Authorship_Attribution/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Clone_detection/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Defect_detection/RNNS
#python RNNS_disturb.py


#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Authorship_Attribution/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Clone_detection/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Defect_detection/RNNS
#python RNNS_disturb.py

#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Authorship_Attribution/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Clone_detection/RNNS
#python RNNS_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Defect_detection/RNNS
#python RNNS_disturb.py




#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Authorship_Attribution/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Clone_detection/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeT5/Defect_detection/ACCENT
#python ACCENT_disturb.py


#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Authorship_Attribution/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Clone_detection/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeGPT/Defect_detection/ACCENT
#python ACCENT_disturb.py
#
#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Authorship_Attribution/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Clone_detection/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/GCBert/Defect_detection/ACCENT
#python ACCENT_disturb.py
#
#
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Authorship_Attribution/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Clone_detection/ACCENT
#python ACCENT_disturb.py
#cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/CodeBert/Defect_detection/ACCENT
#python ACCENT_disturb.py


# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/transfer
#python transfer.py

#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/transfer
#python transfer.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/transfer
#python transfer.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/transfer
#python transfer.py
#
#
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/Alert
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/Alert
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/Alert
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/Alert
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/Alert
#python Transfer_feature.py
#
#
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/Carrot
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/Carrot
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/Carrot
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/Carrot
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/Carrot
#python Transfer_feature.py
#
#
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/RNNS
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/RNNS
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/RNNS
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/RNNS
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/RNNS
#python Transfer_feature.py
#

## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/MHM
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/MHM
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/MHM
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/MHM
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/MHM
#python Transfer_feature.py
#
#
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/WIR
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/WIR
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/WIR
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/WIR
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/WIR
#python Transfer_feature.py


## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
## - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - *
## * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/ACCENT
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/ACCENT
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/ACCENT
#python Transfer_feature.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/ACCENT
#python Transfer_feature.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/ACCENT
#python Transfer_feature.py
#

cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/Ours/Experiments/TDAA
python TDAA.py

#cd /root/autodl-tmp/Experiments/Transferability/WN_Models/Authorship_Attribution/datasets
#python near_k_net.py
#cd /root/autodl-tmp/Experiments/Transferability/WN_Models/Clone_detection/datasets
#python near_k_net.py
#cd /root/autodl-tmp/Experiments/Transferability/WN_Models/Defect_detection/datasets
#python near_k_net.py



#cd /root/autodl-tmp/Experiments/Transferability/TF_IDF/Authorship_Attribution
#python train_tfidf.py
#cd /root/autodl-tmp/Experiments/Transferability/TF_IDF/Clone_detection
#python train_tfidf.py
#cd /root/autodl-tmp/Experiments/Transferability/TF_IDF/Defect_detection
#python train_tfidf.py


#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/ITGen
#python run_itgen.py

#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/ITGen
#python run_itgen.py


#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/ITGen
#python run_itgen.py


#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/ITGen
#python run_itgen.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/ITGen
#python run_itgen.py




#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Authorship_Attribution/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Clone_detection/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeBert/Defect_detection/ITGen
#python convert_jsonl_to_csv.py
#
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Authorship_Attribution/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Clone_detection/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/GCBert/Defect_detection/ITGen
#python convert_jsonl_to_csv.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Authorship_Attribution/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Clone_detection/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeT5/Defect_detection/ITGen
#python convert_jsonl_to_csv.py
#
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Authorship_Attribution/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Clone_detection/ITGen
#python convert_jsonl_to_csv.py
#cd /root/autodl-tmp/Experiments/Transferability/CodeGPT/Defect_detection/ITGen
#python convert_jsonl_to_csv.py



#cd /root/autodl-tmp/Experiments/Transferability/Ours/Experiments/transfer
#python transfer.py
