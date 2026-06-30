import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def code_tokenizer(text):
    """自定义代码分词器，处理特殊符号和单词"""
    return re.findall(r'\w+|[^\w\s]+', text)


def safe_polyfit(x, y, degree=1):
    """简化版安全计算多项式拟合"""
    # 检查数据是否全相同
    if np.all(y == y[0]):
        return np.array([0.0, y[0]])  # 返回水平线
        # 检查数据点是否足够
    if len(x) < 2:
        return np.array([0.0, np.mean(y)])  # 返回常数线
    # 正常计算
    return np.polyfit(x, y, degree)


save_path = "disturb_with_disturb/"
os.makedirs(save_path, exist_ok=True)

data = np.load('../features_disturbed.npz')
clean_features = data['clean_features']
disturb_features = data['disturb_features']
ranks = data['ranks']

all_similarities = []
slopes = []  # +++ 初始化斜率存储列表 +++

unique_ranks = np.unique(ranks)

for rank in unique_ranks:
    # 筛选当前Rank数据
    mask = (ranks == rank)
    current_clean = clean_features[mask]
    current_disturb = disturb_features[mask]

    if len(current_clean) == 0:
        print(f"Rank {rank} 没有可用数据，跳过...")
        continue

    n_steps = len(current_disturb)
    similarities = np.zeros(n_steps)

    # 计算初始相似度（扰动特征0 vs 干净特征）
    if n_steps >= 1:
        similarities[0] = cosine_similarity(
            current_clean[0].reshape(1, -1),
            current_disturb[0].reshape(1, -1)
        )[0][0]

    # 计算后续相似度（扰动特征i vs 扰动特征i-1）
    if n_steps > 1:
        prev_disturb = current_disturb[:-1]  # 前N-1个特征
        next_disturb = current_disturb[1:]  # 后N-1个特征
        similarities[1:] = cosine_similarity(prev_disturb, next_disturb).diagonal()

    x_values = np.arange(len(similarities))

    all_similarities.extend(similarities)
    # 添加趋势线
    if len(x_values) >= 2:  # +++ 添加数据有效性检查 +++
        trend_coeff = safe_polyfit(x_values, similarities, 1)
        trend_line = np.poly1d(trend_coeff)
        slopes.append(trend_coeff[0])  # +++ 收集斜率值 +++
        plt.plot(x_values, trend_line(x_values),
                 color='#d95f02',
                 linestyle='--',
                 linewidth=1.5,
                 label='Trend Line')
    else:
        print(f"Rank {rank} 数据点不足，无法计算趋势线")

    # 可视化部分保持不变
    plt.figure(figsize=(12, 6))

    plt.plot(x_values, similarities,
             marker='o',
             linestyle='--',
             color='#2c7fb8',
             linewidth=2,
             markersize=0.5,
             markeredgewidth=0,
             alpha=0.7)

    avg_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    plt.title(f'Feature Cosine Similarity With Last Progression (Rank {rank})\n'
              f'Average: {avg_sim:.4f} ± {std_sim:.4f}',
              fontsize=14,
              pad=20)
    plt.xlabel('Perturbation Step', fontsize=12, labelpad=10)
    plt.ylabel('Feature Cosine Similarity', fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.ylim(-0.68, 1.05)

    trend_coeff = safe_polyfit(x_values, similarities, 1)
    trend_line = np.poly1d(trend_coeff)
    plt.plot(x_values, trend_line(x_values),
             color='#d95f02',
             linestyle='--',
             linewidth=1.5,
             label='Trend Line')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'rank_{rank}_feature_similarity_trend.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close()

print("所有Rank处理完成！")

# === 新增：全局统计 ===
print("\n全局相似度统计:")
print(f"最大值: {np.max(all_similarities):.4f}")
print(f"最小值: {np.min(all_similarities):.4f}")
print(f"平均值: {np.mean(all_similarities):.4f}")
print(f"标准差: {np.std(all_similarities):.4f}")

# === 新增：全局斜率统计 ===
print("\n全局斜率统计:")
print(f"平均斜率: {np.mean(slopes):.4f}")
print(f"斜率标准差: {np.std(slopes):.4f}")
print(f"最大斜率: {np.max(slopes):.4f}")
print(f"最小斜率: {np.min(slopes):.4f}")

# 全局相似度统计:
# 最大值: 1.0000
# 最小值: 0.3201
# 平均值: 0.9750
# 标准差: 0.0465
#
# 全局斜率统计:
# 平均斜率: 0.0009
# 斜率标准差: 0.0377
# 最大斜率: 0.2134
# 最小斜率: -0.2518
