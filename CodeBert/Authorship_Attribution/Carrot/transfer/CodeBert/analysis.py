# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeBert \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 31
# 平均相似度: 0.9366
# 最小相似度: 0.8158
# 最大相似度: 0.9997
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 21
# 平均相似度: 0.9636
# 最小相似度: 0.7761
# 最大相似度: 0.9982
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeBert/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 31
# 平均距离: 0.3110
# 最小距离: 0.0227
# 最大距离: 0.6070
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 21
# 平均距离: 0.2168
# 最小距离: 0.0592
# 最大距离: 0.6692
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.8138
# 特征最小相似度: 0.6774
# 特征最大相似度: 0.9298
#
# 迁移失败样本:
# 特征平均相似度: 0.9295
# 特征最小相似度: 0.8402
# 特征最大相似度: 0.9916
# 迁移成功样本:
# 特征平均L2距离: 12.0524
# 特征最小L2距离: 7.7656
# 特征最大L2距离: 16.4251
#
# 迁移失败样本:
# 特征平均L2距离: 7.6148
# 特征最小L2距离: 2.6911
# 特征最大L2距离: 11.5468
