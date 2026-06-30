# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 82
# 平均相似度: 0.9046
# 最小相似度: 0.6965
# 最大相似度: 0.9976
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 3
# 平均相似度: 0.9387
# 最小相似度: 0.8838
# 最大相似度: 0.9681
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 82
# 平均距离: 0.3974
# 最小距离: 0.0694
# 最大距离: 0.7791
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 3
# 平均距离: 0.3341
# 最小距离: 0.2527
# 最大距离: 0.4821
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5053
# 特征最小相似度: 0.4782
# 特征最大相似度: 0.5324
#
# 迁移失败样本:
# 特征平均相似度: 0.9294
# 特征最小相似度: 0.6080
# 特征最大相似度: 0.9972
# 迁移成功样本:
# 特征平均L2距离: 19.5289
# 特征最小L2距离: 19.0284
# 特征最大L2距离: 20.0294
#
# 迁移失败样本:
# 特征平均L2距离: 6.3024
# 特征最小L2距离: 1.4833
# 特征最大L2距离: 17.0360
