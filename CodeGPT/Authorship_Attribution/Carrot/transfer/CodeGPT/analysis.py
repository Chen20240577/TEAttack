# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeGPT \
    --victim CodeGPT \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 21
# 平均相似度: 0.8846
# 最小相似度: 0.6992
# 最大相似度: 0.9751
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 22
# 平均相似度: 0.9080
# 最小相似度: 0.7293
# 最大相似度: 0.9677
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeGPT/Carrot_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 21
# 平均距离: 0.4560
# 最小距离: 0.2232
# 最大距离: 0.7756
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 22
# 平均距离: 0.4111
# 最小距离: 0.2542
# 最大距离: 0.7358
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.8838
# 特征最小相似度: -0.0152
# 特征最大相似度: 1.0000
#
# 迁移失败样本:
# 特征平均相似度: 0.9997
# 特征最小相似度: 0.9967
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 12.8859
# 特征最小L2距离: 0.3467
# 特征最大L2距离: 79.0544
#
# 迁移失败样本:
# 特征平均L2距离: 0.7272
# 特征最小L2距离: 0.2659
# 特征最大L2距离: 2.9458
