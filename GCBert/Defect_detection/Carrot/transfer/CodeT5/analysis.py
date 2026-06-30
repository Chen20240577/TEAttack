# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model GCBert \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 6
# 平均相似度: 0.8435
# 最小相似度: 0.7770
# 最大相似度: 0.8962
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 18
# 平均相似度: 0.8091
# 最小相似度: 0.5150
# 最大相似度: 0.9999
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/GCBert/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 6
# 平均距离: 0.5552
# 最小距离: 0.4556
# 最大距离: 0.6679
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 18
# 平均距离: 0.5660
# 最小距离: 0.0173
# 最大距离: 0.9849
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
