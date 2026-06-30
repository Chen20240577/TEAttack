#!/bin/bash
# nohup ./run_scripts.sh > ../logs/run_scripts.log 2>&1 &
# tail -f ../logs/run_scripts.log
# chmod +x run_scripts.sh

python fine-tuning.py
