# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_migration_similarities():
    # 加载特征数据
    data = np.load('./transfer_features.npz')
    clean_features = data['clean_features']
    attack_features = data['attack_features']
    success_flags = data['success_flags']

    # 计算每个样本对的余弦相似度
    paired_similarities = np.diag(cosine_similarity(clean_features, attack_features))

    # 根据迁移成功标志分组
    success_similarities = paired_similarities[success_flags]
    failure_similarities = paired_similarities[~success_flags]

    # 迁移成功样本统计
    if len(success_similarities) > 0:
        print("迁移成功样本:")
        print(f"特征平均相似度: {success_similarities.mean():.4f}")
        print(f"特征最小相似度: {success_similarities.min():.4f}")
        print(f"特征最大相似度: {success_similarities.max():.4f}")
    else:
        print("迁移成功样本: 无数据")

    print()

    # 迁移失败样本统计
    if len(failure_similarities) > 0:
        print("迁移失败样本:")
        print(f"特征平均相似度: {failure_similarities.mean():.4f}")
        print(f"特征最小相似度: {failure_similarities.min():.4f}")
        print(f"特征最大相似度: {failure_similarities.max():.4f}")
    else:
        print("迁移失败样本: 无数据")


if __name__ == '__main__':
    calculate_migration_similarities()

# 迁移成功样本:
# 特征平均相似度: 0.5199
# 特征最小相似度: 0.1476
# 特征最大相似度: 0.9488
#
# 迁移失败样本:
# 特征平均相似度: 0.6490
# 特征最小相似度: 0.3889
# 特征最大相似度: 0.9524
