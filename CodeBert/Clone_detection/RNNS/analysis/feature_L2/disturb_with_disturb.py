import os
import re

import matplotlib.pyplot as plt
import numpy as np


def code_tokenizer(text):
    """自定义代码分词器，处理特殊符号和单词"""
    return re.findall(r'\w+|[^\w\s]+', text)


save_path = "disturb_with_disturb/"
os.makedirs(save_path, exist_ok=True)

data = np.load('../features_disturbed.npz')
clean_features = data['clean_features']
disturb_features = data['disturb_features']
ranks = data['ranks']

all_distances = []
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
    distances = np.zeros(n_steps)

    # 计算初始距离（扰动特征0 vs 干净特征）
    if n_steps >= 1:
        distances[0] = np.linalg.norm(current_clean[0] - current_disturb[0])

    # 计算后续距离（扰动特征i vs 扰动特征i-1）
    if n_steps > 1:
        prev_disturb = current_disturb[:-1]  # 前N-1个特征
        next_disturb = current_disturb[1:]  # 后N-1个特征
        distances[1:] = np.linalg.norm(prev_disturb - next_disturb, axis=1)

    # 可视化部分
    plt.figure(figsize=(12, 6))
    x_values = np.arange(len(distances))

    all_distances.extend(distances)
    # 添加趋势线
    if len(x_values) >= 2:  # +++ 添加数据有效性检查 +++
        trend_coeff = np.polyfit(x_values, distances, 1)
        trend_line = np.poly1d(trend_coeff)
        slopes.append(trend_coeff[0])  # +++ 收集斜率值 +++
        plt.plot(x_values, trend_line(x_values),
                 color='#d95f02',
                 linestyle='--',
                 linewidth=1.5,
                 label='Trend Line')
    else:
        print(f"Rank {rank} 数据点不足，无法计算趋势线")

    plt.plot(x_values, distances,
             marker='o',
             linestyle='--',
             color='#2c7fb8',
             linewidth=2,
             markersize=0.5,
             markeredgewidth=0,
             alpha=0.7)

    avg_dist = np.mean(distances)
    std_dist = np.std(distances)

    plt.title(f'Feature L2 Distance With Last Progression (Rank {rank})\n'
              f'Average: {avg_dist:.4f} ± {std_dist:.4f}',
              fontsize=14,
              pad=20)
    plt.xlabel('Perturbation Step', fontsize=12, labelpad=10)
    plt.ylabel('Feature L2 Distance', fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle=':')
    # 动态调整Y轴范围
    plt.ylim(0, 41.1)

    trend_coeff = np.polyfit(x_values, distances, 1)
    trend_line = np.poly1d(trend_coeff)
    plt.plot(x_values, trend_line(x_values),
             color='#d95f02',
             linestyle='--',
             linewidth=1.5,
             label='Trend Line')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'rank_{rank}_feature_L2_distance_trend.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close()

print("所有Rank处理完成！")

# === 新增：全局统计 ===
print("\n全局距离统计:")
print(f"最大值: {np.max(all_distances):.4f}")
print(f"最小值: {np.min(all_distances):.4f}")
print(f"平均值: {np.mean(all_distances):.4f}")
print(f"标准差: {np.std(all_distances):.4f}")
# === 新增：全局斜率统计 ===
print("\n全局斜率统计:")
print(f"平均斜率: {np.mean(slopes):.4f}")
print(f"斜率标准差: {np.std(slopes):.4f}")
print(f"最大斜率: {np.max(slopes):.4f}")
print(f"最小斜率: {np.min(slopes):.4f}")

# 全局距离统计:
# 最大值: 36.5143
# 最小值: 0.0000
# 平均值: 1.8506
# 标准差: 5.6437
#
# 全局斜率统计:
# 平均斜率: 0.0092
# 斜率标准差: 0.0379
# 最大斜率: 0.2171
# 最小斜率: -0.1671
