#!/bin/bash
# nohup ./run_scripts.sh > ./run_scripts.log 2>&1 &
# tail -f ./run_scripts.log
# chmod +x run_scripts.sh

python vocab_embedding.py

cd datasets
python extract_varname.py
python extract_formalparater.py
python near_k_voc.py

