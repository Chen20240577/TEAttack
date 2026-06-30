# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_similarities():
    sns.set(style="whitegrid")

    # 1. 数据读取
    df = pd.read_csv('origin_with_adv/updated_file_with_similarities.csv')
    sim_matrix = pd.read_csv('origin_with_adv/original_feature_similarity_matrix.csv', index_col=0)

    # ======== 2. 热力图单独存储 ========
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="coolwarm", annot=False, xticklabels=False, yticklabels=False)

    # 计算上三角的平均值，排除对角线
    avg_sim = np.mean(sim_matrix.values[np.triu_indices_from(sim_matrix.values, k=1)])
    # 🔖 在左上角添加平均值注释
    plt.text(0.5, 0.5, f"Avg Sim: {avg_sim:.3f}", color='black',
             fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))

    plt.title("Original Code feature Similarity Heatmap")
    plt.tight_layout()
    plt.savefig("./origin_with_adv/original_feature_similarity_heatmap.png")
    plt.close()

    avg_pairwise_sim = df['Similarity'].mean()

    # ======== 3. 对抗样本相似度柱状图 ========
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df['Similarity'])
    plt.axhline(avg_pairwise_sim, color='red', linestyle='--', label=f'Avg = {avg_pairwise_sim:.3f}')
    plt.xlabel("Sample Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Original vs Adversarial Feature Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./origin_with_adv/original_vs_adversarial_feature_similarity.png")
    plt.close()

    # ======== 4. 低相似度样本提示 ========
    threshold = 0.9
    low_sim_df = df[df['Similarity'] < threshold]
    print(f"\n🔍 Number of low-similarity samples (< {threshold}): {len(low_sim_df)}")
    if not low_sim_df.empty:
        print(low_sim_df[['Original_Code', 'Adversarial_Code', 'Similarity']])


if __name__ == '__main__':
    visualize_similarities()
