# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 3
# 平均相似度: 0.8146
# 最小相似度: 0.7189
# 最大相似度: 0.9851
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 56
# 平均相似度: 0.8677
# 最小相似度: 0.5897
# 最大相似度: 0.9985
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 3
# 平均距离: 0.5479
# 最小距离: 0.1726
# 最大距离: 0.7498
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 56
# 平均距离: 0.4645
# 最小距离: 0.0545
# 最大距离: 0.9059
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5031
# 特征最小相似度: 0.1476
# 特征最大相似度: 0.8540
#
# 迁移失败样本:
# 特征平均相似度: 0.5970
# 特征最小相似度: 0.2159
# 特征最大相似度: 0.9524
# 迁移成功样本:
# 特征平均L2距离: 20.5097
# 特征最小L2距离: 11.2829
# 特征最大L2距离: 27.4270
#
# 迁移失败样本:
# 特征平均L2距离: 18.0399
# 特征最小L2距离: 6.4498
# 特征最大L2距离: 26.4450
