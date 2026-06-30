# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarities():
    # 加载特征数据
    data = np.load('../features_adv.npz')
    clean_features = data['clean_features']
    adv_features = data['adv_features']
    ranks = data['ranks']

    # 计算原始特征间的相似度矩阵
    print("正在计算原始特征相似度矩阵...")
    original_sim_matrix = cosine_similarity(clean_features)

    # 保存原始特征相似度矩阵
    pd.DataFrame(original_sim_matrix).to_csv(
        './origin_with_adv/original_feature_similarity_matrix.csv',
        index=False,
        float_format='%.4f'
    )
    print("原始特征相似度矩阵已保存")

    # 计算对抗样本相似度
    print("\n正在计算对抗样本相似度...")
    paired_similarities = np.diag(cosine_similarity(clean_features, adv_features))

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Rank': ranks,
        'Original_Code': [f"Sample_{r}" for r in ranks],  # 可根据实际需求替换真实代码
        'Adversarial_Code': [f"AdvSample_{r}" for r in ranks],
        'Similarity': paired_similarities
    })

    # 保存对抗样本相似度结果
    result_df.to_csv('./origin_with_adv/updated_file_with_similarities.csv',
                     index=False,
                     float_format='%.4f')
    print("对抗样本相似度结果已保存")

    # 创建一个掩码矩阵，对角线元素为True
    diag_mask = np.eye(original_sim_matrix.shape[0], dtype=bool)
    # 获取非对角线元素
    non_similarities = original_sim_matrix[~diag_mask]
    # 打印统计信息
    print("\n原始特征相似度：")
    print(f"平均距离: {non_similarities.mean():.4f}")  # 修改统计描述
    print(f"最小距离: {non_similarities.min():.4f}")
    print(f"最大距离: {non_similarities.max():.4f}")

    # 打印统计信息
    print("\n对抗样本特征相似度：")
    print(f"平均距离: {paired_similarities.mean():.4f}")  # 修改统计描述
    print(f"最小距离: {paired_similarities.min():.4f}")
    print(f"最大距离: {paired_similarities.max():.4f}")


if __name__ == '__main__':
    calculate_similarities()

# python cos_sim.py
# python visualize_similarity.py
# python disturb_with_origin.py
# python disturb_with_disturb.py

# 原始特征相似度：
# 平均距离: 0.8359
# 最小距离: -0.1276
# 最大距离: 0.9910
#
# 对抗样本特征相似度：
# 平均距离: 0.9247
# 最小距离: 0.0807
# 最大距离: 0.9994
