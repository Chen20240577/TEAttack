# -*- coding: utf-8 -*-
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from code_tokenizer import code_tokenizer


def load_codes(paths):
    codes = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip('\n')  # 移除行尾换行符
                parts = line.split("<CODESPLIT>")
                if len(parts) >= 2:
                    code = parts[0]  # 提取代码部分
                    # 将字面转义序列（如 \n）转换为实际字符
                    code = code.encode().decode('unicode_escape')
                    codes.append(code)
                else:
                    # 可选的错误处理：跳过格式不正确的行
                    print(f"Warning: 行格式错误，已跳过 - {line[:100]}")
    return codes


def main():
    # ===== 你的全部语料（⚠️ 一定要覆盖训练 + 推理分布）=====
    paths = [
        "../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/train.txt",
        "../../Datasets/Authorship_Attribution/codebert-mlm/processed_gcjpy/valid.txt",
    ]

    print("Loading codes...")
    codes = load_codes(paths)
    print(f"Total codes: {len(codes)}")

    print("Training TF-IDF...")
    vectorizer = TfidfVectorizer(
        tokenizer=code_tokenizer,
        lowercase=False,
        max_features=1000
    )
    vectorizer.fit(codes)

    print("Saving tfidf.pkl ...")
    joblib.dump(vectorizer, "tfidf.pkl")

    print("Done.")


if __name__ == "__main__":
    main()
