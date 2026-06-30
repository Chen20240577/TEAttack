# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 118
# 平均相似度: 0.4455
# 最小相似度: 0.0977
# 最大相似度: 0.8025
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 19
# 平均相似度: 0.3784
# 最小相似度: 0.2053
# 最大相似度: 0.5406
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 118
# 平均距离: 1.0404
# 最小距离: 0.6285
# 最大距离: 1.3433
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 19
# 平均距离: 1.1119
# 最小距离: 0.9586
# 最大距离: 1.2607
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.6893
# 特征最小相似度: 0.0427
# 特征最大相似度: 1.0000
#
# 迁移失败样本:
# 特征平均相似度: 0.7469
# 特征最小相似度: 0.7042
# 特征最大相似度: 0.7895
# 迁移成功样本:
# 特征平均L2距离: 10.7496
# 特征最小L2距离: 0.0031
# 特征最大L2距离: 29.1110
#
# 迁移失败样本:
# 特征平均L2距离: 14.9704
# 特征最小L2距离: 13.7412
# 特征最大L2距离: 16.1996
