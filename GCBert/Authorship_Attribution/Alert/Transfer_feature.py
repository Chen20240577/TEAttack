# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim CodeBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim CodeT5 \
    --method Alert")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim CodeGPT \
    --method Alert")
