import ast
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances


# 自定义分词器，处理代码中的单词和符号
def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


# 读取CSV文件
df = pd.read_csv('../../../../../AdvExamples/Clone_detection/Alert_Bert.csv')
df['Original Code'] = df['Original Code'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

# 清洗空值，确保所有文本都是字符串
df = df.dropna(subset=['Original Code', 'Adversarial Code'])  # 去除空值行

df['Original Code'] = df['Original Code'].apply(lambda x: x[3] if isinstance(x, list) and len(x) > 2 else x).astype(str)
df['Adversarial Code'] = df['Adversarial Code'].astype(str)

# 提取Original Code和Adversarial Code的文本
original_corpus = df['Original Code'].tolist()
adversarial_corpus = df['Adversarial Code'].tolist()

# 初始化TF-IDF向量化器，使用自定义分词器
vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
original_vectors = vectorizer.fit_transform(original_corpus)
adversarial_vectors = vectorizer.transform(adversarial_corpus)

# 计算Original Code之间的L2距离矩阵
original_L2 = euclidean_distances(original_vectors)
original_L2_df = pd.DataFrame(original_L2, index=df['Index'], columns=df['Index'])

# 计算每个Original与对应Adversarial的余弦相似度
df['Original_Adversarial_L2_distance'] = [
    euclidean_distances(original_vectors[i], adversarial_vectors[i])[0][0]
    for i in range(len(df))
]

# 保存结果
original_L2_df.to_csv('./origin_with_adv/original_L2_distance_matrix.csv')
df.to_csv('./origin_with_adv/updated_file_with_L2_distance.csv', index=False)
# 创建一个掩码矩阵，对角线元素为True
diag_mask = np.eye(original_L2.shape[0], dtype=bool)
# 获取非对角线元素
non_diag_distances = original_L2[~diag_mask]
# 打印统计信息
print("\n原始样本L2距离：")
print(f"平均距离: {non_diag_distances.mean():.4f}")  # 修改统计描述
print(f"最小距离: {non_diag_distances.min():.4f}")
print(f"最大距离: {non_diag_distances.max():.4f}")

# 打印统计信息
print("\n对抗样本L2距离：")
print(f"平均距离: {df['Original_Adversarial_L2_distance'].mean():.4f}")  # 修改统计描述
print(f"最小距离: {df['Original_Adversarial_L2_distance'].min():.4f}")
print(f"最大距离: {df['Original_Adversarial_L2_distance'].max():.4f}")

# 原始样本L2距离：
# 平均距离: 1.2687
# 最小距离: 0.0000
# 最大距离: 1.3819
#
# 对抗样本L2距离：
# 平均距离: 1.2065
# 最小距离: 0.3521
# 最大距离: 1.3751

# python L2_distance.py
# python visualize_distance.py
# python disturb_with_origin.py
# python disturb_with_disturb.py
#
