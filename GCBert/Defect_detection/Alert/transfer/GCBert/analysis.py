# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 136
# 平均相似度: 0.8120
# 最小相似度: 0.3685
# 最大相似度: 0.9992
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 44
# 平均相似度: 0.8214
# 最小相似度: 0.4129
# 最大相似度: 0.9992
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 136
# 平均距离: 0.5414
# 最小距离: 0.0397
# 最大距离: 1.1238
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 44
# 平均距离: 0.5325
# 最小距离: 0.0396
# 最大距离: 1.0836
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.7534
# 特征最小相似度: 0.0332
# 特征最大相似度: 0.9914
#
# 迁移失败样本:
# 特征平均相似度: 0.8917
# 特征最小相似度: 0.3791
# 特征最大相似度: 0.9993
# 迁移成功样本:
# 特征平均L2距离: 10.0484
# 特征最小L2距离: 1.4344
# 特征最大L2距离: 26.0860
#
# 迁移失败样本:
# 特征平均L2距离: 6.2570
# 特征最小L2距离: 0.4237
# 特征最大L2距离: 22.0324
