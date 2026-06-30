# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 22
# 平均相似度: 0.4537
# 最小相似度: 0.2520
# 最大相似度: 0.8144
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 22
# 平均距离: 1.0359
# 最小距离: 0.6093
# 最大距离: 1.2231
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本: 无数据
#
# 迁移失败样本:
# 特征平均相似度: 0.8826
# 特征最小相似度: 0.0993
# 特征最大相似度: 0.9994
# 迁移成功样本: 无数据
#
# 迁移失败样本:
# 特征平均L2距离: 48.8419
# 特征最小L2距离: 5.7071
# 特征最大L2距离: 205.7213
