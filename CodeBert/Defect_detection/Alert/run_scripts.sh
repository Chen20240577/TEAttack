#!/bin/bash
# nohup ./run_scripts.sh > ../logs/run_scripts.log 2>&1 &
# tail -f ../logs/run_scripts.log

python GA_Attack_disturb.py
sleep 5
nvidia-smi | grep 'python' | awk '{print $5}' | xargs -r -I{} kill -9 {}

python feature_adv_get.py
sleep 5
nvidia-smi | grep 'python' | awk '{print $5}' | xargs -r -I{} kill -9 {}

python feature_disturb_get.py
