# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim CodeBert \
    --method Carrot")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim GCBert \
    --method Carrot")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim CodeT5 \
    --method Carrot")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")
