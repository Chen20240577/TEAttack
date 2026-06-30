# -*- coding: utf-8 -*-
import os

os.system("python cos_sim_transfer.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_cos_transfer.py")

os.system("python L2_distance_transfer.py \
    --task Clone_detection \
    --model CodeBert \
    --victim CodeGPT \
    --method Alert")

os.system("python feature_L2_transfer.py")

# 任务: Clone_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeBert/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_similarities.csv
#
# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 34
# 平均相似度: 0.4659
# 最小相似度: 0.1826
# 最大相似度: 0.7885
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 2
# 平均相似度: 0.5889
# 最小相似度: 0.4500
# 最大相似度: 0.7278
#
# 结果已保存到: ./transfer_with_adv_similarities.csv
# 任务: Clone_detection
# 受害者模型: CodeGPT (文件缩写: GPT)
# 方法: Alert
# 输入文件: ../../../../../TransfAEs/Clone_detection/CodeBert/Alert_GPT_transfer.csv
# 输出文件: ./transfer_with_adv_L2_distance.csv
#
# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 34
# 平均距离: 1.0177
# 最小距离: 0.6503
# 最大距离: 1.2786
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 2
# 平均距离: 0.8933
# 最小距离: 0.7379
# 最大距离: 1.0488
#
# 结果已保存到: ./transfer_with_adv_L2_distance.csv
# 迁移成功样本:
# 特征平均相似度: 0.5031
# 特征最小相似度: 0.1476
# 特征最大相似度: 0.8540
#
# 迁移失败样本:
# 特征平均相似度: 0.5970
# 特征最小相似度: 0.2159
# 特征最大相似度: 0.9524
# 迁移成功样本:
# 特征平均L2距离: 20.5097
# 特征最小L2距离: 11.2829
# 特征最大L2距离: 27.4270
#
# 迁移失败样本:
# 特征平均L2距离: 18.0399
# 特征最小L2距离: 6.4498
# 特征最大L2距离: 26.4450
