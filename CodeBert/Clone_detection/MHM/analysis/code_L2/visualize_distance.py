# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")

# ======== 1. 读取数据 ========
df = pd.read_csv('./origin_with_adv/updated_file_with_L2_distance.csv')
sim_matrix = pd.read_csv('./origin_with_adv/original_L2_distance_matrix.csv', index_col=0)

# ======== 2. 相似度热力图（Original 之间） ========
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap="coolwarm", annot=False, xticklabels=False, yticklabels=False)

# 计算上三角的平均值，排除对角线
avg_L2 = np.mean(sim_matrix.values[np.triu_indices_from(sim_matrix.values, k=1)])
# 🔖 在左上角添加平均值注释
plt.text(0.5, 0.5, f"Avg L2: {avg_L2:.3f}", color='black',
         fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))

plt.title("Original Code L2 Distance Heatmap")
plt.tight_layout()
plt.savefig("./origin_with_adv/original_L2_distance_heatmap.png")
plt.close()
avg_pairwise_sim = df['Original_Adversarial_L2_distance'].mean()

# ======== 3. 原始与对抗样本的相似度柱状图 ========
plt.figure(figsize=(12, 6))
plt.bar(range(len(df)), df['Original_Adversarial_L2_distance'])
plt.axhline(avg_pairwise_sim, color='red', linestyle='--', label=f'Avg = {avg_pairwise_sim:.3f}')
plt.xlabel("Sample Index")
plt.ylabel("L2 Distance")
plt.title("Original vs Adversarial Code L2_distance")
plt.legend()
plt.tight_layout()
plt.savefig("./origin_with_adv/original_vs_adversarial_L2_distance.png")
plt.close()

# ======== 4. 高L2距离样本提示 ========
threshold = 1.0
high_L2_df = df[df['Original_Adversarial_L2_distance'] > threshold]
print(f"\n🔍 Number of high-L2_distance samples (> {threshold}): {len(high_L2_df)}")
if not high_L2_df.empty:
    print(high_L2_df[['Original Code', 'Adversarial Code', 'Original_Adversarial_L2_distance']])
