# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 55
# 平均相似度: 0.9169
# 最小相似度: 0.7127
# 最大相似度: 0.9737
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 55
# 平均距离: 0.3940
# 最小距离: 0.2294
# 最大距离: 0.7580
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5610
# 特征最小相似度: 0.1870
# 特征最大相似度: 0.7745
#
# 迁移失败样本:
# 特征平均相似度: 0.6614
# 特征最小相似度: 0.2981
# 特征最大相似度: 0.9465
# 迁移成功样本:
# 特征平均L2距离: 155.9836
# 特征最小L2距离: 112.8987
# 特征最大L2距离: 215.1781
#
# 迁移失败样本:
# 特征平均L2距离: 132.6949
# 特征最小L2距离: 55.2574
# 特征最大L2距离: 199.5618
