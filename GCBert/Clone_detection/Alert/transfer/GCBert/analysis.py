# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model GCBert \
    --victim GCBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 7
# 平均相似度: 0.5399
# 最小相似度: 0.4353
# 最大相似度: 0.6525
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: GCBert (文件缩写: GCBert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/GCBert/Alert_GCBert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 7
# 平均距离: 0.9548
# 最小距离: 0.8336
# 最大距离: 1.0627
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本: 无数据
#
# 迁移失败样本:
# 特征平均相似度: 0.9498
# 特征最小相似度: 0.5618
# 特征最大相似度: 1.0000
# 迁移成功样本: 无数据
#
# 迁移失败样本:
# 特征平均L2距离: 3.8510
# 特征最小L2距离: 0.1458
# 特征最大L2距离: 19.2784
