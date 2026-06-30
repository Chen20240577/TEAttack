# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('../')
from load import load

tasks = ['authorship', 'clone', 'defect']
models = ['bert', 'gcbert', 't5', 'gpt2']
# tasks = ['defect']
# models = ['gcbert']

# attackers = ['TEAA_greedy', 'TEAA_broad', 'TEAA_reverse', 'TEAA_wordnet', 'TEAA_subs',
# 'MHM_thld_100', 'MHM_thld_050', 'MHM_thld_000', 'TEAA_num',
# 'Alert_GA_random_thld_000', 'Alert_GA_random_thld_100',
# 'Alert_GR_random_thld_000','Alert_GR_random_thld_100',
# 'Alert_GA_word2vec_thld_000', 'Alert_GA_word2vec_thld_100',
# 'Alert_GA_wordnet_thld_000', 'Alert_GA_wordnet_thld_100',
# 'Alert_GA_model_thld_000', 'Alert_GA_model_thld_100',]
attackers = ['Alert_GA_model_thld_000', 'Alert_GA_model_thld_100']

attack_type = {
    'TEAA_greedy': 'TEAA_greedy',
    'TEAA_reverse': 'TEAA_greedy',
    'TEAA_broad': 'TEAA_broad',
    'TEAA_wordnet': 'TEAA_wordnet',
    'TEAA_subs': 'TEAA_subs',
    "MHM_thld_100": "MHM_thld",
    "MHM_thld_050": "MHM_thld",
    "MHM_thld_000": "MHM_thld",
    "TEAA_num": "TEAA_num",
    'Alert_GA_random_thld_000':'Alert_GA_thld',
    'Alert_GA_random_thld_100':'Alert_GA_thld',
    'Alert_GA_word2vec_thld_000':'Alert_GA_thld',
    'Alert_GA_word2vec_thld_100':'Alert_GA_thld',
    'Alert_GA_wordnet_thld_000':'Alert_GA_thld',
    'Alert_GA_wordnet_thld_100':'Alert_GA_thld',
    'Alert_GA_model_thld_000':'Alert_GA_thld',
    'Alert_GA_model_thld_100':'Alert_GA_thld',
    'Alert_GR_random_thld_000':'Alert_GR_thld',
    'Alert_GR_random_thld_100':'Alert_GR_thld',
}

K = 30
# 替换词的数量（比如word2vec模型生成的替换词数量）
for attacker in attackers:
    if attacker == 'TEAA_reverse':
        reverse = True
    else:
        reverse = False
    thld = 0.000

    if attacker == 'MHM_thld_050':
        thld = 0.050
    elif attacker in ['MHM_thld_100',
                      'Alert_GA_random_thld_100',
                      'Alert_GA_word2vec_thld_100',
                      'Alert_GA_wordnet_thld_100',
                      'Alert_GA_model_thld_100',
                      'Alert_GR_random_thld_100',]:
        thld = 0.100

    for model in models:
        store_name = load.store_name[model]

        for task in tasks:
            task_name = load.task_name_get(task)
            language_type = load.languages[task]
            log_name = './logs/' + task + "_" + attacker + '_' + model + '.log'
            number_label = load.number_labels[task]
            date_file = load.date_file_get(task, model)
            batch_size = load.batch_size_get(task, model)
            nearest_k_path = load.nearest_k_path_get(task, K, attack_type[attacker])
            adv_store_path = "../AEs/" + task_name + "/" + attacker + "_" + store_name + ".csv"

            subs = None
            if 'random' in attacker:
                subs = 'random'
            elif 'word2vec' in attacker:
                subs = 'word2vec'
            elif 'wordnet' in attacker:
                subs = 'wordnet'
            elif 'model' in attacker:
                subs = 'model'

            os.system("CUDA_VISIBLE_DEVICES=0 python ./attack.py \
                --data_file={} \
                --task={}\
                --model_type={} \
                --checkpoint_prefix=saved_models/best_acc_model/model.bin \
                --language_type {} \
                --number_labels {} \
                --batch_size {} \
                --nearest_k_path {}\
                --adv_store_path={} \
                --reverse {}\
                --attacker {}\
                --attack_type {}\
                --subs {}\
                --K {}\
                --thld {}\
                --seed 123456 2>&1 | tee {}".format(date_file,
                                                    task,
                                                    model,
                                                    language_type,
                                                    number_label,
                                                    batch_size,
                                                    nearest_k_path,
                                                    adv_store_path,
                                                    reverse,
                                                    attacker,
                                                    attack_type[attacker],
                                                    subs,
                                                    K,
                                                    thld,
                                                    log_name,
                                                    )
                      )
