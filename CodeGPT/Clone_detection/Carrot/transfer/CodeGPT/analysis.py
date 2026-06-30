# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 4
# 平均相似度: 0.3784
# 最小相似度: 0.0000
# 最大相似度: 0.9294
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 4
# 平均距离: 1.0339
# 最小距离: 0.3758
# 最大距离: 1.4142
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.0036
# 特征最小相似度: -0.5233
# 特征最大相似度: 0.9980
#
# 迁移失败样本: 无数据
# 迁移成功样本:
# 特征平均L2距离: 115.9959
# 特征最小L2距离: 3.9796
# 特征最大L2距离: 165.5653
#
# 迁移失败样本: 无数据
