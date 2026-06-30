# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances  # 修改导入的度量方式


def calculate_L2_distance():
    # 加载特征数据
    data = np.load('../features_adv.npz')
    clean_features = data['clean_features']
    adv_features = data['adv_features']
    ranks = data['ranks']

    # 计算原始特征间的距离矩阵（使用欧氏距离）
    print("正在计算原始特征距离矩阵...")
    original_dist_matrix = euclidean_distances(clean_features)  # 修改为欧氏距离

    # 保存原始特征距离矩阵
    pd.DataFrame(original_dist_matrix).to_csv(
        './origin_with_adv/original_feature_L2_distance_matrix.csv',
        index=False,
        float_format='%.4f'
    )
    print("原始特征L2距离矩阵已保存")

    # 计算对抗样本距离（使用二范数）
    print("\n正在计算对抗样本L2距离...")
    paired_distances = np.linalg.norm(clean_features - adv_features, axis=1)  # 直接计算每对的欧氏距离

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Rank': ranks,
        'Original_Code': [f"Sample_{r}" for r in ranks],
        'Adversarial_Code': [f"AdvSample_{r}" for r in ranks],
        'Distance': paired_distances  # 列名改为Distance
    })

    # 保存对抗样本距离结果
    result_df.to_csv('./origin_with_adv/updated_file_with_L2_distance.csv',
                     index=False,
                     float_format='%.4f')
    print("对抗样本L2距离结果已保存")

    # 创建一个掩码矩阵，对角线元素为True
    diag_mask = np.eye(original_dist_matrix.shape[0], dtype=bool)
    # 获取非对角线元素
    non_diag_distances = original_dist_matrix[~diag_mask]
    # 打印统计信息
    print("\n原始特征L2距离：")
    print(f"平均距离: {non_diag_distances.mean():.4f}")  # 修改统计描述
    print(f"最小距离: {non_diag_distances.min():.4f}")
    print(f"最大距离: {non_diag_distances.max():.4f}")

    # 打印统计信息
    print("\n对抗样本L2距离：")
    print(f"平均距离: {paired_distances.mean():.4f}")  # 修改统计描述
    print(f"最小距离: {paired_distances.min():.4f}")
    print(f"最大距离: {paired_distances.max():.4f}")


if __name__ == '__main__':
    calculate_L2_distance()

# python L2_distance.py
# python visualize_distance.py
# python disturb_with_origin.py
# python disturb_with_disturb.py

# 原始特征L2距离：
# 平均距离: 108.6251
# 最小距离: 0.0000
# 最大距离: 289.2296
#
# 对抗样本L2距离：
# 平均距离: 72.6521
# 最小距离: 0.1406
# 最大距离: 253.4811
