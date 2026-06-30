# -*- coding: utf-8 -*-
import os

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeBert \
    --method ITGen")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim GCBert \
    --method ITGen")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeT5 \
    --method ITGen")

os.system("CUDA_VISIBLE_DEVICES=0 python feature_transfer_get.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeGPT \
    --method ITGen")
