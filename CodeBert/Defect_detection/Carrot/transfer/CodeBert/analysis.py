# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Defect_detection \
    --model CodeBert \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Defect_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeBert/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 70
# 平均相似度: 0.8291
# 最小相似度: 0.2656
# 最大相似度: 0.9994
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 0
# 平均相似度: nan
# 最小相似度: nan
# 最大相似度: nan
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Defect_detection
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Defect_detection/CodeBert/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 70
# 平均距离: 0.5262
# 最小距离: 0.0352
# 最大距离: 1.2120
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 0
# 平均距离: nan
# 最小距离: nan
# 最大距离: nan
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.8564
# 特征最小相似度: 0.2084
# 特征最大相似度: 0.9952
#
# 迁移失败样本:
# 特征平均相似度: 0.9739
# 特征最小相似度: 0.7729
# 特征最大相似度: 1.0000
# 迁移成功样本:
# 特征平均L2距离: 9.0641
# 特征最小L2距离: 1.4940
# 特征最大L2距离: 27.9115
#
# 迁移失败样本:
# 特征平均L2距离: 3.8075
# 特征最小L2距离: 0.0865
# 特征最大L2距离: 14.1295
