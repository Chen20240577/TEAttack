# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model CodeT5 \
    --victim CodeT5 \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 38
# 平均相似度: 0.8016
# 最小相似度: 0.3758
# 最大相似度: 0.9998
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeT5 (文件缩写: T5)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeT5/Carrot_T5_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 38
# 平均距离: 0.5931
# 最小距离: 0.0184
# 最大距离: 1.1173
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.9983
# 特征最小相似度: 0.9962
# 特征最大相似度: 1.0000
#
# 迁移失败样本:
# 特征平均相似度: 0.9985
# 特征最小相似度: 0.9878
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 6.0231
# 特征最小L2距离: 0.3062
# 特征最大L2距离: 10.3349
#
# 迁移失败样本:
# 特征平均L2距离: 5.5010
# 特征最小L2距离: 0.3183
# 特征最大L2距离: 18.4045
