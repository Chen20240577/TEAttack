# -*- coding: utf-8 -*-
import numpy as np


def calculate_L2_distance():
    # 加载特征数据
    data = np.load('./transfer_features.npz')
    clean_features = data['clean_features']
    attack_features = data['attack_features']
    success_flags = data['success_flags']

    # 计算每个样本对的欧氏距离
    paired_distances = np.linalg.norm(clean_features - attack_features, axis=1)

    # 根据迁移成功标志分组
    success_distances = paired_distances[success_flags]
    failure_distances = paired_distances[~success_flags]

    # 迁移成功样本统计
    if len(success_distances) > 0:
        print("迁移成功样本:")
        print(f"特征平均L2距离: {success_distances.mean():.4f}")
        print(f"特征最小L2距离: {success_distances.min():.4f}")
        print(f"特征最大L2距离: {success_distances.max():.4f}")
    else:
        print("迁移成功样本: 无数据")

    print()

    # 迁移失败样本统计
    if len(failure_distances) > 0:
        print("迁移失败样本:")
        print(f"特征平均L2距离: {failure_distances.mean():.4f}")
        print(f"特征最小L2距离: {failure_distances.min():.4f}")
        print(f"特征最大L2距离: {failure_distances.max():.4f}")
    else:
        print("迁移失败样本: 无数据")


if __name__ == '__main__':
    calculate_L2_distance()

