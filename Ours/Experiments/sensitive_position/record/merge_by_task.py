# -*- coding: utf-8 -*-
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def merge_csv_by_index_and_columns(directory_path, output_dir="merged_results"):
    """
    基于Index列匹配，并确保匹配行的前三列完全一致后才合并CSV文件

    Parameters:
    directory_path (str): 包含CSV文件的目录路径
    output_dir (str): 合并后文件输出目录
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 按task分组文件
    task_groups = defaultdict(list)

    for file_path in csv_files:
        # 从文件名提取task和model
        filename = os.path.basename(file_path)
        if '_' not in filename or not filename.endswith('.csv'):
            print(f"跳过不符合命名规范的文件: {filename}")
            continue

        # 解析文件名：task_model.csv
        name_parts = filename.replace('.csv', '').split('_')
        if len(name_parts) < 2:
            print(f"文件名格式错误: {filename}")
            continue

        task = name_parts[0]
        model = '_'.join(name_parts[1:])

        task_groups[task].append((model, file_path))

    print(f"按task分组完成，找到 {len(task_groups)} 个不同的task")

    merge_results = {}

    # 处理每个task的文件
    for task, files in task_groups.items():
        print(f"\n🔧 处理task: {task}, 包含 {len(files)} 个文件")

        if len(files) < 2:
            print(f"task {task} 只有1个文件，跳过合并")
            continue

        # 读取所有文件
        file_data = {}
        for model, file_path in files:
            try:
                df = pd.read_csv(file_path)
                # 确保有Index列
                if 'Index' not in df.columns:
                    print(f"文件 {file_path} 没有Index列，跳过")
                    continue

                # 🔥 新增：跳过第三列为空的行 [2,5](@ref)
                if df.shape[1] >= 3:  # 确保至少有3列
                    third_col_name = df.columns[2]
                    before_count = len(df)
                    # 删除第三列为空的行 [2](@ref)
                    df = df.dropna(subset=[third_col_name])
                    after_count = len(df)
                    removed_count = before_count - after_count
                    if removed_count > 0:
                        print(f"  ⚠️  移除 {model} 文件中第三列为空的行: {removed_count} 行")

                file_data[model] = df
                print(f"  ✅ 读取 {model}: {df.shape[0]}行, {df.shape[1]}列")
            except Exception as e:
                print(f"  ❌ 读取 {file_path} 时出错: {e}")
                continue

        if len(file_data) < 2:
            print(f"task {task} 有效文件不足2个，跳过")
            continue

        # 选择第一个文件作为基准
        base_model, base_df = next(iter(file_data.items()))
        print(f"  选择 {base_model} 作为基准文件")

        # 创建合并结果DataFrame
        merged_df = base_df.iloc[:, :4].copy() if base_df.shape[1] >= 4 else base_df.iloc[:, :3].copy()
        if base_df.shape[1] >= 4:
            merged_df.rename(columns={merged_df.columns[3]: f"{base_model} Score"}, inplace=True)

        # 记录合并统计信息
        merge_stats = {
            'total_rows': len(base_df),
            'successful_merges': 0,
            'failed_merges': 0,
            'missing_indices': defaultdict(list),
            'column_mismatches': defaultdict(list),
            'skipped_third_column_empty': 0  # 🔥 新增：记录跳过的行数
        }

        # 合并其他文件的Score列
        for model, df in file_data.items():
            if model == base_model:
                continue

            print(f"  🔄 合并 {model} 的Score列...")

            # 临时存储合并结果
            temp_scores = []

            for idx, base_row in base_df.iterrows():
                base_index = base_row['Index']

                # 🔥 新增：检查第三列是否为空，如果为空则跳过 [3,5](@ref)
                if pd.isna(base_row.iloc[2]) or base_row.iloc[2] == '':
                    temp_scores.append(np.nan)
                    merge_stats['skipped_third_column_empty'] += 1
                    merge_stats['failed_merges'] += 1
                    continue

                # 在目标文件中查找相同Index的行
                target_rows = df[df['Index'] == base_index]

                if len(target_rows) == 0:
                    # Index不存在
                    temp_scores.append(np.nan)
                    merge_stats['missing_indices'][model].append(base_index)
                    merge_stats['failed_merges'] += 1

                elif len(target_rows) > 0:
                    target_row = target_rows.iloc[0]

                    # 检查前三列是否一致
                    base_first_three = base_row.iloc[:3]  # 前三列数据
                    target_first_three = target_row.iloc[:3]

                    # 🔥 新增：检查目标文件第三列是否为空 [3](@ref)
                    if pd.isna(target_row.iloc[2]) or target_row.iloc[2] == '':
                        temp_scores.append(np.nan)
                        merge_stats['failed_merges'] += 1
                        print(f"    ⚠️ Index {base_index} 的目标文件第三列为空，跳过")
                    elif base_first_three.equals(target_first_three):
                        # 前三列一致，合并Score列
                        if target_row.shape[0] >= 4:  # 确保有第四列
                            temp_scores.append(target_row.iloc[3])
                            merge_stats['successful_merges'] += 1
                        else:
                            temp_scores.append(np.nan)
                            merge_stats['failed_merges'] += 1
                    else:
                        # 前三列不一致
                        temp_scores.append(np.nan)
                        merge_stats['column_mismatches'][model].append(base_index)
                        merge_stats['failed_merges'] += 1
                        print(f"    ⚠️ Index {base_index} 的前三列不一致")
                else:
                    temp_scores.append(np.nan)
                    merge_stats['failed_merges'] += 1

            # 添加合并后的Score列
            score_col_name = f"{model} Score"
            merged_df[score_col_name] = temp_scores

            print(
                f"    📊 {model} 合并完成: 成功 {len([x for x in temp_scores if not pd.isna(x)])}/{len(temp_scores)} 行")

        # 保存合并结果
        output_file = os.path.join(output_dir, f"{task}_merged.csv")
        merged_df.to_csv(output_file, index=False)

        # 计算合并成功率
        total_attempts = merge_stats['successful_merges'] + merge_stats['failed_merges']
        success_rate = (merge_stats['successful_merges'] / total_attempts * 100) if total_attempts > 0 else 0

        print(f"  ✅ 合并完成: {output_file}")
        print(f"  📈 合并成功率: {success_rate:.1f}%")
        print(
            f"  📊 总行数: {merge_stats['total_rows']}, 成功: {merge_stats['successful_merges']}, 失败: {merge_stats['failed_merges']}")
        if merge_stats['skipped_third_column_empty'] > 0:
            print(f"  🚫 因第三列为空跳过的行: {merge_stats['skipped_third_column_empty']}")

        # 记录结果
        merge_results[task] = {
            'output_file': output_file,
            'stats': merge_stats,
            'merged_columns': list(merged_df.columns)
        }

    # 生成合并报告
    generate_merge_report(merge_results, output_dir)

    return merge_results


def generate_merge_report(merge_results, output_dir):
    """生成详细的合并报告"""
    report_file = os.path.join(output_dir, "merge_report.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("CSV文件合并报告\n")
        f.write("=" * 50 + "\n\n")

        for task, result in merge_results.items():
            f.write(f"任务: {task}\n")
            f.write(f"输出文件: {result['output_file']}\n")
            f.write(f"合并列: {result['merged_columns']}\n")

            stats = result['stats']
            total_attempts = stats['successful_merges'] + stats['failed_merges']
            success_rate = (stats['successful_merges'] / total_attempts * 100) if total_attempts > 0 else 0

            f.write(f"合并统计:\n")
            f.write(f"  - 总行数: {stats['total_rows']}\n")
            f.write(f"  - 成功合并: {stats['successful_merges']}\n")
            f.write(f"  - 合并失败: {stats['failed_merges']}\n")
            f.write(f"  - 成功率: {success_rate:.1f}%\n")

            # 🔥 新增：报告跳过的行数
            if stats['skipped_third_column_empty'] > 0:
                f.write(f"  - 因第三列为空跳过的行: {stats['skipped_third_column_empty']}\n")

            if stats['missing_indices']:
                f.write(f"缺失Index统计:\n")
                for model, indices in stats['missing_indices'].items():
                    f.write(f"  - {model}: {len(indices)} 个Index缺失\n")

            if stats['column_mismatches']:
                f.write(f"列不匹配统计:\n")
                for model, indices in stats['column_mismatches'].items():
                    f.write(f"  - {model}: {len(indices)} 处前三列不匹配\n")

            f.write("\n" + "-" * 50 + "\n\n")

    print(f"📄 详细合并报告已保存至: {report_file}")


# 使用示例
if __name__ == "__main__":
    # 设置包含CSV文件的目录路径
    input_directory = "."  # 当前目录，可以修改为您的CSV文件所在目录
    output_directory = "../merged_record"

    print("开始基于Index和前三列一致性的CSV文件合并...")
    print("=" * 60)

    # 方法1: 精确匹配合并
    results = merge_csv_by_index_and_columns(input_directory, output_directory)

    print("\n" + "=" * 60)
    print("合并操作完成！")
