import json
import logging
import os

import nltk
from gensim.models.word2vec import Word2Vec

nltk.data.path.append('/root/nltk_data')

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess_code(code):
    """预处理代码：替换特殊字符"""
    return code.replace("\\n", "\n").replace('\"', '"')


class CorpusIterator:
    """迭代器类，用于高效读取和处理语料库文件"""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        js = json.loads(line)
                        code = js['func']  # 获取代码内容
                        # 预处理代码
                        code = preprocess_code(code)
                        # 将代码分割成token列表
                        yield code.split()
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"处理JSON行时出错: {e}, 行内容: {line}")
                        continue


def train_vocab(corpus_files, model_dir, vector_size=64):
    """
    训练Word2Vec模型

    参数:
    corpus_files: 语料库文件路径列表
    model_dir: 模型保存目录
    vector_size: 词向量维度
    """
    # 创建目录（如果不存在）
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'node_w2v_code_{vector_size}.model')

    # 检查文件是否存在
    for file_path in corpus_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"语料库文件不存在: {file_path}")

    print(f'Starting training with vector_size={vector_size}')
    print(f'使用语料库文件: {corpus_files}')

    # 创建语料库迭代器
    corpus = CorpusIterator(corpus_files)

    # 训练模型 - 使用 size 参数而不是 vector_size
    model = Word2Vec(
        sentences=corpus,
        size=vector_size,  # 使用 size 而不是 vector_size
        workers=16,
        sg=1,  # skip-gram
        min_count=3,
    )

    # 保存模型
    model.save(model_path)
    print(f'Model saved to {model_path}')

    # 获取词汇量
    vocab_size = len(model.wv.vocab)
    print(f'Vocabulary size: {vocab_size}')

    # 检查是否有 get_latest_training_loss 方法
    if hasattr(model, 'get_latest_training_loss'):
        print(f'Final training loss: {model.get_latest_training_loss()}')
    else:
        print('Training loss information not available in this gensim version')

    return model_path, vocab_size


if __name__ == '__main__':
    print('Starting word2vec training...')

    # 在main函数中定义所有路径
    BASE_DIR = '../../Datasets/Clone_detection/codebert-mlm'

    # 定义语料库文件路径
    corpus_files = [
        os.path.join(BASE_DIR, 'data_folder/data.jsonl'),
    ]

    # 定义模型保存目录
    model_dir = './word2vec_model'

    # 调用训练函数
    model_path, vocab_size = train_vocab(
        corpus_files=corpus_files,
        model_dir=model_dir,
        vector_size=64
    )

    print(f'训练完成! 模型保存至: {model_path}')
    print(f'词汇表大小: {vocab_size}')
