# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeT5 \
    --victim CodeBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 18
# 平均相似度: 0.4897
# 最小相似度: 0.2434
# 最大相似度: 0.7323
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeT5/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 18
# 平均距离: 1.0033
# 最小距离: 0.7317
# 最大距离: 1.2301
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5439
# 特征最小相似度: 0.1957
# 特征最大相似度: 0.9247
#
# 迁移失败样本:
# 特征平均相似度: 0.9299
# 特征最小相似度: 0.4825
# 特征最大相似度: 0.9998
# 迁移成功样本:
# 特征平均L2距离: 18.7917
# 特征最小L2距离: 8.2221
# 特征最大L2距离: 27.0864
#
# 迁移失败样本:
# 特征平均L2距离: 5.8591
# 特征最小L2距离: 0.4177
# 特征最大L2距离: 21.1740
