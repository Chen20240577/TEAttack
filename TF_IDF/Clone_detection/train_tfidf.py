# -*- coding: utf-8 -*-
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from code_tokenizer import code_tokenizer


def load_codes(paths):
    codes = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "func" in obj:
                        codes.append(obj["func"])
                except Exception:
                    pass
    return codes


def main():
    # ===== 你的全部语料（⚠️ 一定要覆盖训练 + 推理分布）=====
    paths = [
        "../../Datasets/Clone_detection/codebert-mlm/data_folder/data.jsonl",
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
