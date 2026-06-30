# -*- coding: utf-8 -*-
import glob
import os
from collections import defaultdict

import pandas as pd


def compare_csv_columns_by_task():
    """
    比较同task的CSV文件的前三列数值是否一致
    文件命名格式: task + '_' + model + '.csv'
    """

    # 获取所有CSV文件
    csv_files = glob.glob("*.csv")
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 按task分组文件
    task_groups = defaultdict(list)

    for file in csv_files:
        # 解析文件名，提取task和model
        filename = os.path.splitext(file)[0]  # 去掉扩展名
        if '_' in filename:
            parts = filename.split('_')
            task = parts[0]
            model = '_'.join(parts[1:])  # 处理可能包含多个下划线的情况
            task_groups[task].append((model, file))

    print(f"按task分组结果: {dict(task_groups)}")

    # 比较每个task组内的文件
    results = {}

    for task, files in task_groups.items():
        print(f"\n=== 比较task: {task} ===")
        print(f"包含文件: {[f[1] for f in files]}")

        if len(files) < 2:
            print(f"task '{task}' 只有一个文件，无需比较")
            results[task] = {"status": "single_file", "message": "只有一个文件，无需比较"}
            continue

        # 读取所有文件的前三列
        dataframes = {}
        consistent = True
        differences = []

        for model, file in files:
            try:
                # 读取CSV文件的前三列，确保数据类型一致[2](@ref)
                df = pd.read_csv(file, usecols=[0, 1, 2], dtype=str, keep_default_na=False)
                dataframes[model] = df
                print(f"  {model}: 读取成功，形状 {df.shape}")
            except Exception as e:
                print(f"  {model}: 读取失败 - {e}")
                consistent = False
                differences.append(f"{model}文件读取错误: {e}")
                continue

        if len(dataframes) < 2:
            print(f"task '{task}' 有效文件不足2个，跳过比较")
            results[task] = {"status": "insufficient_files", "message": "有效文件不足2个"}
            continue

        # 获取第一个DataFrame作为基准
        base_model, base_df = next(iter(dataframes.items()))

        # 比较每个DataFrame与基准是否一致[1,5](@ref)
        for model, df in dataframes.items():
            if model == base_model:
                continue

            print(f"  比较 {base_model} 与 {model}...")

            # 检查形状是否一致[5](@ref)
            if base_df.shape != df.shape:
                print(f"    形状不同: {base_model}{base_df.shape} vs {model}{df.shape}")
                consistent = False
                differences.append(f"{base_model}与{model}形状不同: {base_df.shape} vs {df.shape}")
                continue

            # 比较前三列数据是否完全一致[1,2](@ref)
            try:
                # 方法1: 使用equals进行精确比较[1](@ref)
                if not base_df.equals(df):
                    print(f"    数据内容不同")
                    consistent = False

                    # 找出具体差异位置[2](@ref)
                    diff_mask = base_df != df
                    diff_indices = diff_mask.any(axis=1)

                    if diff_indices.any():
                        diff_rows = base_df[diff_indices]
                        print(f"    发现 {diff_indices.sum()} 行数据不同")

                        for idx in diff_rows.index:
                            for col in range(3):
                                if diff_mask.iloc[idx, col]:
                                    differences.append(
                                        f"行{idx + 1} 第{col + 1}列: "
                                        f"{base_model}='{base_df.iloc[idx, col]}' vs "
                                        f"{model}='{df.iloc[idx, col]}'"
                                    )
                else:
                    print(f"    前三列数据完全一致")

            except Exception as e:
                print(f"    比较过程中出错: {e}")
                consistent = False
                differences.append(f"{base_model}与{model}比较错误: {e}")

        # 记录结果
        if consistent:
            print(f"✓ task '{task}' 所有文件的前三列完全一致")
            results[task] = {"status": "consistent", "message": "所有文件前三列完全一致"}
        else:
            print(f"✗ task '{task}' 存在不一致")
            results[task] = {
                "status": "inconsistent",
                "message": "文件间存在差异",
                "differences": differences
            }

    return results


def generate_summary_report(results):
    """生成比较结果摘要报告"""
    print("\n" + "=" * 50)
    print("比较结果摘要")
    print("=" * 50)

    consistent_tasks = [task for task, result in results.items() if result["status"] == "consistent"]
    inconsistent_tasks = [task for task, result in results.items() if result["status"] == "inconsistent"]
    single_file_tasks = [task for task, result in results.items() if result["status"] == "single_file"]

    print(f"一致的任务数量: {len(consistent_tasks)}")
    print(f"不一致的任务数量: {len(inconsistent_tasks)}")
    print(f"单文件任务数量: {len(single_file_tasks)}")

    if inconsistent_tasks:
        print(f"\n不一致的任务: {inconsistent_tasks}")
        for task in inconsistent_tasks:
            print(f"\n任务 '{task}' 的差异详情:")
            for diff in results[task].get("differences", []):
                print(f"  - {diff}")


# 执行比较
if __name__ == "__main__":
    print("开始比较CSV文件前三列数据...")
    results = compare_csv_columns_by_task()
    generate_summary_report(results)
