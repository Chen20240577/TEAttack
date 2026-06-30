# -*- coding: utf-8 -*-

import os
import sys

sys.path.append('../')
from load import load

tasks = ['authorship', 'clone', 'defect']
Target_Models = ['bert', 'gcbert', 't5', 'gpt2']
Models = ['bert', 'gcbert', 't5', 'gpt2']
# tasks = ['clone', 'defect']
# Target_Models = ['bert']
# Models = ['bert']
# attackers = ['TEAA_broad', 'TEAA_reverse', 'TEAA_wordnet', 'TEAA_subs',
# 'MHM_thld_050', 'MHM_thld_000', 'MHM_thld_100', 'ITGen', 'TEAA_num']
# attackers = ['Alert', 'Carrot', 'ACCENT', 'MHM', 'RNNS', 'WIR']
# attackers = ['Alert_test', 'Alert_diff', 'Alert_sim']
# attackers = ['Alert_GA_random_thld_000', 'Alert_GA_random_thld_100','Alert_GR_random_thld_100',
# 'Alert_GA_word2vec_thld_100','Alert_GA_wordnet_thld_100','Alert_GA_model_thld_100', 'Alert_GA_model_thld_000']

attackers = ['GA_random_30_thld_000', 'GA_random_100_thld_000', 'GA_random_200_thld_000',
             'GA_random_500_thld_000', 'GA_random_2000_thld_000']
# attackers = ['GA_random_5000_thld_000']

paths_front = {
    "ACCENT": "../../../AdvExamples/",
    "Alert": "../../../AdvExamples/",
    "Carrot": "../../../AdvExamples/",
    "MHM": "../../../AdvExamples/",
    "RNNS": "../../../AdvExamples/",
    "WIR": "../../../AdvExamples/",
    "TEAA_reverse": "../AEs/",
    "TEAA_broad": "../AEs/",
    "TEAA_wordnet": "../AEs/",
    "TEAA_subs": "../AEs/",
    "MHM_thld_100":"../AEs/",
    "MHM_thld_050": "../AEs/",
    "MHM_thld_000": "../AEs/",
    "ITGen": "../../../AdvExamples/",
    "Alert_test": "../../../AdvExamples/",
    "Alert_diff": "../../../AdvExamples/",
    "Alert_sim": "../../../AdvExamples/",
    "TEAA_num":"../AEs/",
    "Alert_GA_random_thld_000":"../AEs/",
    "Alert_GA_random_thld_100":"../AEs/",
    "Alert_GR_random_thld_100":"../AEs/",
    "Alert_GA_word2vec_thld_100":"../AEs/",
    "Alert_GA_wordnet_thld_100":"../AEs/",
    "Alert_GA_model_thld_100":"../AEs/",
    "Alert_GA_model_thld_000":"../AEs/",
    "GA_random_30_thld_000":"../AEs/",
    "GA_random_100_thld_000":"../AEs/",
    "GA_random_200_thld_000":"../AEs/",
    "GA_random_500_thld_000":"../AEs/",
    "GA_random_2000_thld_000":"../AEs/",
    "GA_random_5000_thld_000":"../AEs/",
}
paths_back = {
    "ACCENT": "ACCENT_",
    "Alert": "Alert_",
    "Carrot": "Carrot_",
    "MHM": "MHM_",
    "RNNS": "RNNS_",
    "WIR": "WIR_",
    "TEAA_reverse": "TEAA_reverse_",
    "TEAA_broad": "TEAA_broad_",
    "TEAA_wordnet": "TEAA_wordnet_",
    "TEAA_subs": "TEAA_subs_",
    "MHM_thld_100": "MHM_thld_100_",
    "MHM_thld_050": "MHM_thld_050_",
    "MHM_thld_000": "MHM_thld_000_",
    "ITGen": "ITGen_",
    "Alert_test": "Alert_test_",
    "Alert_diff": "Alert_diff_",
    "Alert_sim": "Alert_sim_",
    "TEAA_num":"TEAA_num_",
    "Alert_GA_random_thld_000":"Alert_GA_random_thld_000_",
    "Alert_GA_random_thld_100":"Alert_GA_random_thld_100_",
    "Alert_GR_random_thld_100":"Alert_GR_random_thld_100_",
    "Alert_GA_word2vec_thld_100":"Alert_GA_word2vec_thld_100_",
    "Alert_GA_wordnet_thld_100":"Alert_GA_wordnet_thld_100_",
    "Alert_GA_model_thld_100":"Alert_GA_model_thld_100_",
    "Alert_GA_model_thld_000":"Alert_GA_model_thld_000_",
    "GA_random_30_thld_000":"GA_random_30_thld_000_",
    "GA_random_100_thld_000":"GA_random_100_thld_000_",
    "GA_random_200_thld_000": "GA_random_200_thld_000_",
    "GA_random_500_thld_000": "GA_random_500_thld_000_",
    "GA_random_2000_thld_000": "GA_random_2000_thld_000_",
    "GA_random_5000_thld_000": "GA_random_5000_thld_000_",
}
transfer_path_front = "./Transfer_AEs/"
log_path_front = "./logs/"
# transfer_path_front = "../../../TransfAEs/"
# log_path_front = "../../../TransfAEs/logs/"

for attack in attackers:
    path_front = paths_front[attack]
    path_back = paths_back[attack]
    for target_model in Target_Models:
        model_name = load.model_name_get(target_model)
        # 对同一个模型，进行循环攻击/迁移验证
        # model_target 代表当前循环的被攻击模型
        for task in tasks:
            task_name = load.task_name_get(task)
            language_type = load.languages[task]
            number_label = load.number_labels[task]
            batch_size = load.batch_size_get(task, target_model)

            for model in Models:
                store_name = load.store_name[model]

                # csv_read_path = "../AEs/" + task_name + "/TEAA_no_threshold_" + store_name + ".csv"
                # transfer_store_path = "./Transfer_AEs/" + task_name + "/" + model_name + "/TEAA_no_threshold_" + store_name + "_transfer.csv"
                # log_path = './logs/' + task_name + "/" + model_name + '/transfer_TEAA_no_threshold_' + store_name + '.log'

                csv_read_path = path_front + task_name + "/" + path_back + store_name + ".csv"
                transfer_store_path = transfer_path_front + task_name + "/" + model_name + "/" + path_back + store_name + "_transfer.csv"
                log_path = log_path_front + task_name + "/" + model_name + '/transfer_' + path_back + store_name + '.log'

                os.system("CUDA_VISIBLE_DEVICES=0 python ./csv_attack.py \
                    --csv_read_path={} \
                    --transfer_store_path={} \
                    --task={}\
                    --model_type={} \
                    --checkpoint_prefix=saved_models/best_acc_model/model.bin \
                    --language_type {} \
                    --number_labels {} \
                    --batch_size {} \
                    --seed 123456 2>&1 | tee {}".format(csv_read_path,
                                                        transfer_store_path,
                                                        task,
                                                        target_model,
                                                        language_type,
                                                        number_label,
                                                        batch_size,
                                                        log_path
                                                        )
                          )

                # os.system("CUDA_VISIBLE_DEVICES=0 python ./csv_attack.py \
                #                     --csv_read_path={} \
                #                     --transfer_store_path={} \
                #                     --task={}\
                #                     --model_type={} \
                #                     --checkpoint_prefix=saved_attack_models/idx_0/model_state.pt \
                #                     --language_type {} \
                #                     --number_labels {} \
                #                     --batch_size {} \
                #                     --seed 123456 2>&1 | tee {}".format(csv_read_path,
                #                                                         transfer_store_path,
                #                                                         task,
                #                                                         target_model,
                #                                                         language_type,
                #                                                         number_label,
                #                                                         batch_size,
                #                                                         log_path
                #                                                         )
                #           )


