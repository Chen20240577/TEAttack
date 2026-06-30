#!/bin/bash
export LD_LIBRARY_PATH="/root/miniconda3/envs/CLH/lib:$LD_LIBRARY_PATH"
# nohup ./run_scripts.sh > ./logs/run_scripts.log 2>&1 &
# tail -f ./logs/run_scripts.log
# chmod +x run_scripts.sh

python transfer.py

