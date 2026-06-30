# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim GCBert \
    --method ACCENT")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim GCBert \
    --method ACCENT")

os.system("python feature_L2_transfer.py")
