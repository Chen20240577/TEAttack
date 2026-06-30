# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 19
# 平均相似度: 0.8783
# 最小相似度: 0.5902
# 最大相似度: 0.9874
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 37
# 平均相似度: 0.8575
# 最小相似度: 0.6124
# 最大相似度: 0.9985
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 19
# 平均距离: 0.4292
# 最小距离: 0.1588
# 最大距离: 0.9053
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 37
# 平均距离: 0.4842
# 最小距离: 0.0551
# 最大距离: 0.8804
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.7269
# 特征最小相似度: 0.2849
# 特征最大相似度: 0.9965
#
# 迁移失败样本:
# 特征平均相似度: 0.8912
# 特征最小相似度: 0.5080
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 36.7124
# 特征最小L2距离: 3.0662
# 特征最大L2距离: 78.0315
#
# 迁移失败样本:
# 特征平均L2距离: 18.1976
# 特征最小L2距离: 0.0747
# 特征最大L2距离: 68.7707
