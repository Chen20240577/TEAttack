#!/bin/bash
# nohup ./run_scripts.sh > ../logs/run_scripts.log 2>&1 &
# tail -f ../logs/run_scripts.log


python GA-Attack_disturb.py

python feature_adv_get.py

python feature_disturb_get.py

