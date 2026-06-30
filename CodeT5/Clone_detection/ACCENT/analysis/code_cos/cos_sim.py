import ast
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 自定义分词器，处理代码中的单词和符号
def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


# 读取CSV文件
df = pd.read_csv('../../../../../AdvExamples/Clone_detection/ACCENT_T5.csv')

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

# 计算Original Code之间的余弦相似度矩阵
original_similarity = cosine_similarity(original_vectors)
original_sim_df = pd.DataFrame(original_similarity, index=df['Index'], columns=df['Index'])

# 计算每个Original与对应Adversarial的余弦相似度
df['Original_Adversarial_Similarity'] = [
    cosine_similarity(original_vectors[i], adversarial_vectors[i])[0][0]
    for i in range(len(df))
]

# 保存结果
original_sim_df.to_csv('./origin_with_adv/original_similarity_matrix.csv')
df.to_csv('./origin_with_adv/updated_file_with_similarities.csv', index=False)
# 创建一个掩码矩阵，对角线元素为True
diag_mask = np.eye(original_similarity.shape[0], dtype=bool)
# 获取非对角线元素
non_similarities = original_similarity[~diag_mask]
# 打印统计信息
print("\nOriginal Code之间：")
print(f"平均相似度: {non_similarities.mean():.4f}")  # 修改统计描述
print(f"最小相似度: {non_similarities.min():.4f}")
print(f"最大相似度: {non_similarities.max():.4f}")

# 打印统计信息
print("\nOriginal与对应Adversarial：")
print(f"平均相似度: {df['Original_Adversarial_Similarity'].mean():.4f}")  # 修改统计描述
print(f"最小相似度: {df['Original_Adversarial_Similarity'].min():.4f}")
print(f"最大相似度: {df['Original_Adversarial_Similarity'].max():.4f}")

# Original Code之间：
# 平均相似度: 0.2187
# 最小相似度: 0.0346
# 最大相似度: 1.0000
#
# Original与对应Adversarial：
# 平均相似度: 0.2648
# 最小相似度: 0.0559
# 最大相似度: 0.7674

# python cos_sim.py
# python visualize_similarity.py
# python disturb_with_origin.py
# python disturb_with_disturb.py
