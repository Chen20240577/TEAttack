# -*- coding: utf-8 -*-
import argparse
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 模型名称到文件缩写的映射
MODEL_ABBREVIATIONS = {
    "CodeBert": "Bert",
    "GCBert": "GCBert",
    "CodeT5": "T5",
    "CodeGPT": "GPT",
}


# 自定义分词器，处理代码中的单词和符号
def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


def get_model_abbreviation(model_name):
    """根据模型名称获取文件中的缩写"""
    return MODEL_ABBREVIATIONS.get(model_name, model_name)  # 如果没有映射，使用原模型名称


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='计算代码迁移相似度')

    ## Required parameters - 必需的路径参数
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="任务名称，如: Authorship_Attribution")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="模型名称，如: CodeBert")
    parser.add_argument("--victim", default=None, type=str, required=True,
                        help="模型名称，如: CodeBert")
    parser.add_argument("--method", default=None, type=str, required=True,
                        help="方法名称，如: ACCENT")

    ## Optional parameters - 可选参数
    parser.add_argument("--base_path", default="../../../../../TransfAEs", type=str,
                        help="基础输入路径，默认: ../../../../../TransfAEs")
    parser.add_argument("--output_csv", default="./transfer_with_adv_similarities.csv", type=str,
                        help="输出CSV文件路径，默认: ./transfer_with_adv_similarities.csv")

    args = parser.parse_args()

    # 获取模型在文件名中的缩写
    model_abbr = get_model_abbreviation(args.victim)

    # 动态构建输入文件路径
    input_file = f"{args.base_path}/{args.task}/{args.model}/{args.method}_{model_abbr}_transfer.csv"

    print(f"任务: {args.task}")
    print(f"受害者模型: {args.victim} (文件缩写: {model_abbr})")
    print(f"方法: {args.method}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {args.output_csv}")

    try:
        # 读取新数据集CSV文件
        df = pd.read_csv(input_file)

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

        # 保存结果到文件
        df.to_csv(args.output_csv, index=False)
        print(f"\n结果已保存到: {args.output_csv}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return


if __name__ == '__main__':
    main()

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
