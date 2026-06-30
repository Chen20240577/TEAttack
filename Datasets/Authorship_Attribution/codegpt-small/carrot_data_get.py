# -*- coding: utf-8 -*-
import json

from tqdm import tqdm


def process_txt_to_jsonl(input_path, output_path):
    samples = []

    # 计算文件行数用于进度条
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 读取原始TXT文件
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            # 跳过空行
            if not line.strip():
                continue

            # 分离代码和标签
            if '<CODESPLIT>' in line:
                parts = line.split('<CODESPLIT>')
                code_str = parts[0].strip()
                label_str = parts[1].strip()

                # 处理可能的多个标签值（只取第一个）
                if ' ' in label_str:
                    label_str = label_str.split(' ')[0]

                # 验证标签是否为有效整数
                try:
                    label_int = int(label_str)
                except ValueError:
                    print(f"\n警告：跳过无效标签行: {line.strip()}")
                    continue

                # 创建样本数据
                sample = {
                    "code": code_str,
                    "label": label_int
                }

                samples.append(sample)

    # 保存为JSONL格式 - 使用json.dump但要确保换行符保留
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for sample in samples:
            # 序列化JSON但不转义特殊字符
            json_str = json.dumps(sample, ensure_ascii=False)
            # 手动处理掉多余的转义
            json_str = json_str.replace(r'\\n', r'\n')  # 将双斜杠n恢复为单斜杠n
            json_str = json_str.replace(r'\\t', r'\t')  # 同样处理制表符
            out_f.write(json_str + '\n')

    print(f"\n成功处理 {len(samples)} 个样本")
    print(f"输出文件已保存至: {output_path}")


if __name__ == "__main__":
    # 参数检查

    input_path = "processed_gcjpy/valid.txt"
    output_path = "processed_gcjpy/valid.jsonl"

    process_txt_to_jsonl(input_path, output_path)
