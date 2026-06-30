# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeGPT \
    --method RNNS")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeGPT \
    --method RNNS")

os.system("python feature_L2_transfer.py")
