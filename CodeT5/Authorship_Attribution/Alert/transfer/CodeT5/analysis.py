# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 24
# 平均相似度: 0.8712
# 最小相似度: 0.6310
# 最大相似度: 0.9719
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 24
# 平均相似度: 0.9018
# 最小相似度: 0.7877
# 最大相似度: 0.9959
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 24
# 平均距离: 0.4886
# 最小距离: 0.2369
# 最大距离: 0.8591
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 24
# 平均距离: 0.4082
# 最小距离: 0.0902
# 最大距离: 0.6516
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.6303
# 特征最小相似度: 0.1498
# 特征最大相似度: 0.9337
#
# 迁移失败样本:
# 特征平均相似度: 0.8716
# 特征最小相似度: 0.3723
# 特征最大相似度: 0.9993
# 迁移成功样本:
# 特征平均L2距离: 133.6955
# 特征最小L2距离: 61.5815
# 特征最大L2距离: 219.0994
#
# 迁移失败样本:
# 特征平均L2距离: 65.5940
# 特征最小L2距离: 6.3623
# 特征最大L2距离: 188.5413
