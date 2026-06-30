# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 42
# 平均相似度: 0.8550
# 最小相似度: 0.3807
# 最大相似度: 0.9941
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 65
# 平均相似度: 0.7657
# 最小相似度: 0.3051
# 最大相似度: 0.9962
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 42
# 平均距离: 0.4779
# 最小距离: 0.1088
# 最大距离: 1.1129
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 65
# 平均距离: 0.6143
# 最小距离: 0.0868
# 最大距离: 1.1789
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.9546
# 特征最小相似度: 0.7336
# 特征最大相似度: 0.9991
#
# 迁移失败样本:
# 特征平均相似度: 0.9869
# 特征最小相似度: 0.9313
# 特征最大相似度: 0.9999
# 迁移成功样本:
# 特征平均L2距离: 4.6722
# 特征最小L2距离: 0.7395
# 特征最大L2距离: 16.6753
#
# 迁移失败样本:
# 特征平均L2距离: 3.0086
# 特征最小L2距离: 0.2928
# 特征最大L2距离: 9.6122
