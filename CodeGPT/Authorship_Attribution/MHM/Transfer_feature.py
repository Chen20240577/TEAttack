# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim GCBert \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeT5 \
    --method MHM")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeGPT \
    --method MHM")
