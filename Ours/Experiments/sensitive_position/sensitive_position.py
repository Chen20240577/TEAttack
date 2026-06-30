# -*- coding: utf-8 -*-
# python sensitive_position.py

# 运行暂时解决不了一个问题，通过这种方式调用的时候，需要把各个run.py里面
# from model import Model
# 修改为
# from .model import Model
# 暂时没有找到通用的办法，改完后会导致原本的攻击方法调用run.py的时候报错
# 我是通过pycharm的 ctrl+shift+R 将全部的from model import Model进行修改实现的，需要跑原本的攻击方法就再改回去

# 找到解决办法了，统一改为：
# try:
#     from model import Model
# except:
#     from .model import Model

import sys
import os
sys.path.append('../')
from load import load

# tasks = ['authorship', 'clone', 'defect']
# models = ['bert', 'gcbert', 't5', 'gpt2']
tasks = ['authorship']
models = ['gcbert']

for model in models:
    model_name = load.model_name_get(model)

    for task in tasks:
        task_name = load.task_name_get(task)
        language_type = load.languages[task]
        log_name = './logs/' + task + '_' + model + '.log'
        number_label = load.number_labels[task]
        importance_store_path = './record/' + task + '_' + model + '.csv'
        date_file = load.date_file_get(task, model)
        batch_size = load.batch_size_get(task, model)

        os.system("CUDA_VISIBLE_DEVICES=0 python ./run.py \
            --data_file={} \
            --task={}\
            --model_type={} \
            --checkpoint_prefix=saved_models/best_acc_model/model.bin \
            --language_type {} \
            --number_labels {} \
            --batch_size {} \
            --importance_store_path={} \
            --seed 123456 2>&1 | tee {}".format(date_file,
                                                task,model,
                                                language_type,
                                                number_label,
                                                batch_size,
                                                importance_store_path,
                                                log_name
                                                )
                  )



