# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim GCBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim GCBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeGPT/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 4
# 平均相似度: 0.6581
# 最小相似度: 0.0544
# 最大相似度: 0.9505
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 1
# 平均相似度: 0.0178
# 最小相似度: 0.0178
# 最大相似度: 0.0178
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeGPT/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 4
# 平均距离: 0.7231
# 最小距离: 0.3147
# 最大距离: 1.3752
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 1
# 平均距离: 1.4015
# 最小距离: 1.4015
# 最大距离: 1.4015
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5131
# 特征最小相似度: 0.3089
# 特征最大相似度: 0.7557
#
# 迁移失败样本:
# 特征平均相似度: 0.5548
# 特征最小相似度: 0.1476
# 特征最大相似度: 0.9524
# 迁移成功样本:
# 特征平均L2距离: 20.5701
# 特征最小L2距离: 14.5184
# 特征最大L2距离: 24.8462
#
# 迁移失败样本:
# 特征平均L2距离: 19.1888
# 特征最小L2距离: 6.4498
# 特征最大L2距离: 27.4270
