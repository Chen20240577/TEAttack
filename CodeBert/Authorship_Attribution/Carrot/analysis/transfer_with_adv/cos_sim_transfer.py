# -*- coding: utf-8 -*-
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 自定义分词器，处理代码中的单词和符号
def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


# 读取新数据集CSV文件
df = pd.read_csv('../../../../../TransfAEs/Authorship_Attribution/CodeBert/Carrot_Bert_transfer.csv')  # 请确认路径是否正确

# 清洗空值，确保所有文本都是字符串
df = df.dropna(subset=['Original Code', 'Disturb Code'])  # 使用新数据集的列名
df['Original Code'] = df['Original Code'].astype(str)
df['Disturb Code'] = df['Disturb Code'].astype(str)

# 提取Original Code和Disturb Code的文本
original_corpus = df['Original Code'].tolist()
disturb_corpus = df['Disturb Code'].tolist()

# 初始化TF-IDF向量化器，使用自定义分词器
vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
original_vectors = vectorizer.fit_transform(original_corpus)
disturb_vectors = vectorizer.transform(disturb_corpus)

# 计算每个Original与对应Disturb的余弦相似度
df['Original_Disturb_Similarity'] = [
    cosine_similarity(original_vectors[i], disturb_vectors[i])[0][0]
    for i in range(len(df))
]

# 根据True label分组计算平均相似度
success_similarities = df[df['True label'] == True]['Original_Disturb_Similarity']
failure_similarities = df[df['True label'] == False]['Original_Disturb_Similarity']

# 打印统计信息
print("\n迁移成功组 - Original Code与Transfer Code的相似度：")
print(f"样本数量: {len(success_similarities)}")
print(f"平均相似度: {success_similarities.mean():.4f}")
print(f"最小相似度: {success_similarities.min():.4f}")

print(f"最大相似度: {success_similarities.max():.4f}")

print("\n迁移失败组 - Original Code与Adversarial Code的相似度：")
print(f"样本数量: {len(failure_similarities)}")
print(f"平均相似度: {failure_similarities.mean():.4f}")
print(f"最小相似度: {failure_similarities.min():.4f}")
print(f"最大相似度: {failure_similarities.max():.4f}")

# 可选：保存结果到文件
df.to_csv('./transfer_with_adv_similarities.csv', index=False)

# 迁移成功组 - Original Code与Transfer Code的相似度：
# 样本数量: 44
# 平均相似度: 0.8586
# 最小相似度: 0.6151
# 最大相似度: 0.9972
#
# 迁移失败组 - Original Code与Adversarial Code的相似度：
# 样本数量: 17
# 平均相似度: 0.8454
# 最小相似度: 0.6837
# 最大相似度: 0.9882
