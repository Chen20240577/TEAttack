# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeBert \
    --method RNNS")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim GCBert \
    --method RNNS")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeT5 \
    --method RNNS")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeGPT \
    --method RNNS")
