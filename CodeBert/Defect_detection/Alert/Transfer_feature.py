# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeBert \
    --victim GCBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeT5 \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeGPT \
    --method Alert")
