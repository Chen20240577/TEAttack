# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model CodeT5 \
    --victim CodeBert \
    --method Carrot")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 5
# 平均相似度: 0.9125
# 最小相似度: 0.8504
# 最大相似度: 0.9965
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 9
# 平均相似度: 0.9708
# 最小相似度: 0.8282
# 最大相似度: 0.9982
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Carrot
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/CodeT5/Carrot_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 5
# 平均距离: 0.3828
# 最小距离: 0.0834
# 最大距离: 0.5471
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 9
# 平均距离: 0.1816
# 最小距离: 0.0605
# 最大距离: 0.5861
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
