# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 28
# 平均相似度: 0.7839
# 最小相似度: 0.3204
# 最大相似度: 0.9919
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 79
# 平均相似度: 0.7772
# 最小相似度: 0.3261
# 最大相似度: 0.9968
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeT5/Alert_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 28
# 平均距离: 0.6149
# 最小距离: 0.1271
# 最大距离: 1.1658
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 79
# 平均距离: 0.6093
# 最小距离: 0.0796
# 最大距离: 1.1610
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.9995
# 特征最小相似度: 0.9984
# 特征最大相似度: 0.9998
#
# 迁移失败样本:
# 特征平均相似度: 0.9995
# 特征最小相似度: 0.9963
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 3.6498
# 特征最小L2距离: 2.2715
# 特征最大L2距离: 6.6342
#
# 迁移失败样本:
# 特征平均L2距离: 3.4061
# 特征最小L2距离: 0.4372
# 特征最大L2距离: 9.3137
