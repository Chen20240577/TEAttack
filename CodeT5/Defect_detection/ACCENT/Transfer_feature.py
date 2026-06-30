# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeBert \
    --method ACCENT")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim GCBert \
    --method ACCENT")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method ACCENT")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeGPT \
    --method ACCENT")
