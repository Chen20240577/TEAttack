# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeGPT \
    --method ITGen")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeGPT \
    --method ITGen")

os.system("python feature_L2_transfer.py")

