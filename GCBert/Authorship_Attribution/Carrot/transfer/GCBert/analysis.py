# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim GCBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim GCBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 29
# 平均相似度: 0.9412
# 最小相似度: 0.8149
# 最大相似度: 0.9991
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 22
# 平均相似度: 0.9434
# 最小相似度: 0.7928
# 最大相似度: 0.9979
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 29
# 平均距离: 0.2996
# 最小距离: 0.0431
# 最大距离: 0.6084
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 22
# 平均距离: 0.2913
# 最小距离: 0.0644
# 最大距离: 0.6438
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.7609
# 特征最小相似度: 0.2557
# 特征最大相似度: 0.9943
#
# 迁移失败样本:
# 特征平均相似度: 0.8828
# 特征最小相似度: 0.3697
# 特征最大相似度: 0.9993
# 迁移成功样本:
# 特征平均L2距离: 12.1453
# 特征最小L2距离: 2.1297
# 特征最大L2距离: 23.0387
#
# 迁移失败样本:
# 特征平均L2距离: 6.7238
# 特征最小L2距离: 0.7428
# 特征最大L2距离: 20.5291
