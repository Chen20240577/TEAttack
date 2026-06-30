import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在

# 读取CSV文件
df = pd.read_csv('../disturbed.csv')

# 数据预处理
df = df.dropna(subset=['Original Code', 'Disturb Code'])
df['Original Code'] = df['Original Code'].astype(str)
df['Disturb Code'] = df['Disturb Code'].astype(str)

all_similarities = []
slopes = []  # +++ 初始化斜率存储列表 +++

# 获取所有唯一的Rank值
unique_ranks = df['Rank'].unique()

for rank in unique_ranks:
    # 筛选当前Rank的数据
    target_df = df[df['Rank'] == rank].copy().reset_index(drop=True)

    if target_df.empty:
        print(f"Rank {rank} 没有可用数据，跳过...")
        continue

    # 获取原始代码和扰动代码列表
    original_code = target_df['Original Code'].iloc[0]
    disturb_texts = target_df['Disturb Code'].tolist()

    # 构建比较序列（前向代码 vs 当前代码）
    forward_texts = [original_code] + disturb_texts[:-1]  # 前向代码序列
    current_texts = disturb_texts  # 当前代码序列

    # 初始化并训练TF-IDF向量化器
    vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
    combined_texts = forward_texts + current_texts
    vectorizer.fit(combined_texts)

    # 转换文本为向量
    forward_vectors = vectorizer.transform(forward_texts)
    current_vectors = vectorizer.transform(current_texts)

    # 批量计算余弦相似度
    sim_matrix = cosine_similarity(forward_vectors, current_vectors)
    similarities = sim_matrix.diagonal().tolist()

    # 添加相似度结果到DataFrame
    target_df['Cosine_Similarity'] = similarities

    all_similarities.extend(similarities)
    # 添加趋势线
    if len(target_df.index) >= 2:  # +++ 添加数据有效性检查 +++
        trend_coeff = safe_polyfit(target_df.index, similarities, 1)
        trend_line = np.poly1d(trend_coeff)
        slopes.append(trend_coeff[0])  # +++ 收集斜率值 +++
        plt.plot(target_df.index, trend_line(target_df.index),
                 color='#d95f02',
                 linestyle='--',
                 linewidth=1.5,
                 label='Trend Line')
    else:
        print(f"Rank {rank} 数据点不足，无法计算趋势线")

    # ========================= 查找最低相似度 =========================
    min_index = target_df['Cosine_Similarity'].idxmin()
    min_row = target_df.iloc[min_index]

    forward_code = forward_texts[min_index]
    current_code = current_texts[min_index]
    # ========================= 绘图部分 =========================
    plt.figure(figsize=(12, 6))
    plt.plot(target_df.index, similarities,
             marker='o',
             linestyle='--',
             color='#2c7fb8',
             linewidth=2,
             markersize=0.5,
             markeredgewidth=0)

    plt.title(f'Cosine Similarity With Last Progression (Rank {rank})\n'
              f'Average: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}',
              fontsize=14)
    plt.xlabel('Perturbation Step', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # 添加趋势线
    z = safe_polyfit(target_df.index, similarities, 1)
    p = np.poly1d(z)
    plt.plot(target_df.index, p(target_df.index),
             color='#d95f02',
             linestyle='--',
             label='Trend Line')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'rank_{rank}_similarity_trend_.png'), dpi=300)
    plt.close()  # 关闭图形释放内存

print("所有Rank处理完成！生成的图片和结果保存在目录:", os.path.abspath(save_path))

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
# 最大值: 0.9998
# 最小值: 0.5438
# 平均值: 0.9277
# 标准差: 0.0915
#
# 全局斜率统计:
# 平均斜率: 0.0024
# 斜率标准差: 0.0196
# 最大斜率: 0.0805
# 最小斜率: -0.0434
