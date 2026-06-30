import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances  # 修改导入


def code_tokenizer(text):
    """自定义代码分词器，处理特殊符号和单词"""
    return re.findall(r'\w+|[^\w\s]+', text)


save_path = "disturb_with_origin/"
os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在

data = np.load('../features_disturbed.npz')
clean_features = data['clean_features']
disturb_features = data['disturb_features']
ranks = data['ranks']

all_distances = []
slopes = []  # +++ 初始化斜率存储列表 +++

# 获取所有唯一的Rank值
unique_ranks = np.unique(ranks)

for rank in unique_ranks:
    # 筛选当前Rank数据
    mask = (ranks == rank)
    current_clean = clean_features[mask]
    current_disturb = disturb_features[mask]

    if len(current_clean) == 0:
        print(f"Rank {rank} 没有可用数据，跳过...")
        continue

    # 批量计算欧氏距离
    distance_matrix = euclidean_distances(current_clean, current_disturb)  # 替换计算函数
    distances = distance_matrix.diagonal()  # 获取对角线距离值

    # 创建可视化图表
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

    # 绘制距离折线
    plt.plot(x_values, distances,
             marker='o',
             linestyle='--',
             color='#2c7fb8',
             linewidth=2,
             markersize=0.5,
             markeredgewidth=0,
             alpha=0.7)

    # 计算统计指标
    avg_dist = np.mean(distances)  # 变量名修改
    std_dist = np.std(distances)  # 变量名修改

    # 添加统计信息和样式
    plt.title(f'Feature L2 Distance Progression (Rank {rank})\n'  # 标题修改
              f'Average: {avg_dist:.4f} ± {std_dist:.4f}',
              fontsize=14,
              pad=20)
    plt.xlabel('Perturbation Step', fontsize=12, labelpad=10)
    plt.ylabel('Feature L2 Distance', fontsize=12, labelpad=10)  # 标签修改
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.ylim(0, 41.1)  # 根据特征尺度调整范围

    # 添加趋势线
    trend_coeff = np.polyfit(x_values, distances, 1)
    trend_line = np.poly1d(trend_coeff)
    plt.plot(x_values, trend_line(x_values),
             color='#d95f02',
             linestyle='--',
             linewidth=1.5,
             label='Trend Line')

    # 优化布局和保存
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

# 最大值: 33.4225
# 最小值: 0.0000
# 平均值: 1.8610
# 标准差: 6.1141
#
# 全局斜率统计:
# 平均斜率: 0.0079
# 斜率标准差: 0.0425
# 最大斜率: 0.4036
# 最小斜率: -0.2301
