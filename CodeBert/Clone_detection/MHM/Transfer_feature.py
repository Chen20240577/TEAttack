# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeBert \
    --victim GCBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeT5 \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeGPT \
    --method MHM")
