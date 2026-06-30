# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim GCBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim GCBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 113
# 平均相似度: 0.8098
# 最小相似度: 0.3445
# 最大相似度: 1.0000
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Carrot_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 113
# 平均距离: 0.5590
# 最小距离: 0.0066
# 最大距离: 1.1450
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.6779
# 特征最小相似度: 0.4281
# 特征最大相似度: 0.9621
#
# 迁移失败样本:
# 特征平均相似度: 0.8755
# 特征最小相似度: 0.5354
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 11.7352
# 特征最小L2距离: 4.0008
# 特征最大L2距离: 18.2265
#
# 迁移失败样本:
# 特征平均L2距离: 6.8439
# 特征最小L2距离: 0.0000
# 特征最大L2距离: 16.7613
