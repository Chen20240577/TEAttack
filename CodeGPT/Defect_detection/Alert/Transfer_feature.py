# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim CodeBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim GCBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim CodeT5 \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Alert")
