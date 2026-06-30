#!/bin/bash
# nohup ./run_scripts.sh > ./logs/run_scripts.log 2>&1 &
# tail -f ./logs/run_scripts.log
# chmod +x run_scripts.sh

python sensitive_position.py

cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/Ours/Experiments/sensitive_position/record
python merge_by_task.py

cd /Se-liuxinwei/Se-chenlinhua/Experiments/Transferability/Ours/Experiments/sensitive_position
python similarity.py
python similarity2.py
