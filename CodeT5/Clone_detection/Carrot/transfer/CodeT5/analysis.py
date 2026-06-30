# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 5
# 平均相似度: 0.7605
# 最小相似度: 0.6423
# 最大相似度: 0.7999
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 5
# 平均距离: 0.6875
# 最小距离: 0.6326
# 最大距离: 0.8458
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5273
# 特征最小相似度: 0.5273
# 特征最大相似度: 0.5273
#
# 迁移失败样本:
# 特征平均相似度: 0.6812
# 特征最小相似度: -0.0650
# 特征最大相似度: 0.9997
# 迁移成功样本:
# 特征平均L2距离: 160.8948
# 特征最小L2距离: 160.8948
# 特征最大L2距离: 160.8948
#
# 迁移失败样本:
# 特征平均L2距离: 101.8209
# 特征最小L2距离: 4.0867
# 特征最大L2距离: 244.1148
