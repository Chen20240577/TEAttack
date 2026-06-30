# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Authorship_Attribution \
    --model GCBert \
    --victim CodeBert \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 16
# 平均相似度: 0.7956
# 最小相似度: 0.6814
# 最大相似度: 0.9730
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 43
# 平均相似度: 0.8719
# 最小相似度: 0.6168
# 最大相似度: 0.9971
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Authorship_Attribution
# 受害者模型: CodeBert (文件缩写: Bert)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Authorship_Attribution/GCBert/Alert_Bert_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 16
# 平均距离: 0.6207
# 最小距离: 0.2324
# 最大距离: 0.7982
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 43
# 平均距离: 0.4505
# 最小距离: 0.0760
# 最大距离: 0.8755
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
