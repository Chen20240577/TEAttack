# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 38
# 平均相似度: 0.8138
# 最小相似度: 0.6344
# 最大相似度: 0.9984
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 33
# 平均相似度: 0.8309
# 最小相似度: 0.4545
# 最大相似度: 0.9999
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 38
# 平均距离: 0.5693
# 最小距离: 0.0564
# 最大距离: 0.8550
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 33
# 平均距离: 0.5194
# 最小距离: 0.0148
# 最大距离: 1.0445
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.8123
# 特征最小相似度: -0.1739
# 特征最大相似度: 1.0000
#
# 迁移失败样本:
# 特征平均相似度: 0.8685
# 特征最小相似度: 0.3678
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 10.8536
# 特征最小L2距离: 0.0589
# 特征最大L2距离: 60.1950
#
# 迁移失败样本:
# 特征平均L2距离: 14.9173
# 特征最小L2距离: 0.0357
# 特征最大L2距离: 58.1254
