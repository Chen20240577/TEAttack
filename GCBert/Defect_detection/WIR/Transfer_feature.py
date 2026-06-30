# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model GCBert \
    --victim CodeBert \
    --method WIR")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model GCBert \
    --victim GCBert \
    --method WIR")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model GCBert \
    --victim CodeT5 \
    --method WIR")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model GCBert \
    --victim CodeGPT \
    --method WIR")
