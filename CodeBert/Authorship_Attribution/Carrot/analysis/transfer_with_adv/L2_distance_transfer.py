import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances


# 自定义分词器，处理代码中的单词和符号
def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


# 读取新数据集CSV文件
df = pd.read_csv('../../../../../TransfAEs/Authorship_Attribution/CodeBert/Carrot_Bert_transfer.csv')

# 清洗空值，确保所有文本都是字符串
df = df.dropna(subset=['Original Code', 'Disturb Code'])
df['Original Code'] = df['Original Code'].astype(str)
df['Disturb Code'] = df['Disturb Code'].astype(str)

# 提取Original Code和Disturb Code的文本
original_corpus = df['Original Code'].tolist()
disturb_corpus = df['Disturb Code'].tolist()

# 初始化TF-IDF向量化器，使用自定义分词器
vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
original_vectors = vectorizer.fit_transform(original_corpus)
disturb_vectors = vectorizer.transform(disturb_corpus)

# 计算每个Original与对应Disturb的L2距离
df['Original_Disturb_L2_distance'] = [
    euclidean_distances(original_vectors[i], disturb_vectors[i])[0][0]
    for i in range(len(df))
]

# 根据True label分组计算L2距离
success_distances = df[df['True label'] == True]['Original_Disturb_L2_distance']
failure_distances = df[df['True label'] == False]['Original_Disturb_L2_distance']

# 打印统计信息
print("\n迁移成功组 - Original Code与Transfer Code的L2距离：")
print(f"样本数量: {len(success_distances)}")
print(f"平均距离: {success_distances.mean():.4f}")
print(f"最小距离: {success_distances.min():.4f}")
print(f"最大距离: {success_distances.max():.4f}")

print("\n迁移失败组 - Original Code与Adversarial Code的L2距离：")
print(f"样本数量: {len(failure_distances)}")
print(f"平均距离: {failure_distances.mean():.4f}")
print(f"最小距离: {failure_distances.min():.4f}")
print(f"最大距离: {failure_distances.max():.4f}")

# 保存结果
df.to_csv('./transfer_with_adv_L2_distance.csv', index=False)

# 迁移成功组 - Original Code与Transfer Code的L2距离：
# 样本数量: 44
# 平均距离: 0.4714
# 最小距离: 0.0742
# 最大距离: 0.8774
#
# 迁移失败组 - Original Code与Adversarial Code的L2距离：
# 样本数量: 17
# 平均距离: 0.5300
# 最小距离: 0.1537
# 最大距离: 0.7954
