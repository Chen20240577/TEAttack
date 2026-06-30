# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim GCBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeGPT \
    --method MHM")
