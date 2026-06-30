# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 15
# 平均相似度: 0.9293
# 最小相似度: 0.8920
# 最大相似度: 0.9685
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 11
# 平均相似度: 0.9125
# 最小相似度: 0.7310
# 最大相似度: 0.9682
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 15
# 平均距离: 0.3704
# 最小距离: 0.2508
# 最大距离: 0.4647
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 11
# 平均距离: 0.3982
# 最小距离: 0.2524
# 最大距离: 0.7334
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5186
# 特征最小相似度: 0.3089
# 特征最大相似度: 0.9488
#
# 迁移失败样本:
# 特征平均相似度: 0.5586
# 特征最小相似度: 0.1476
# 特征最大相似度: 0.9524
# 迁移成功样本:
# 特征平均L2距离: 20.1421
# 特征最小L2距离: 6.5866
# 特征最大L2距离: 24.8462
#
# 迁移失败样本:
# 特征平均L2距离: 18.9449
# 特征最小L2距离: 6.4498
# 特征最大L2距离: 27.4270
