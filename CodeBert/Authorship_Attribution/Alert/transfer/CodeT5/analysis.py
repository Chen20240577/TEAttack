# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 3
# 平均相似度: 0.8522
# 最小相似度: 0.7905
# 最大相似度: 0.9146
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 54
# 平均相似度: 0.8836
# 最小相似度: 0.6246
# 最大相似度: 0.9958
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 3
# 平均距离: 0.5352
# 最小距离: 0.4133
# 最大距离: 0.6474
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 54
# 平均距离: 0.4470
# 最小距离: 0.0920
# 最大距离: 0.8665
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
