import csv
import os
import re
from collections import defaultdict


def parse_log(log_file_path):
    """解析日志文件，将同一任务的多个部分合并为一条记录"""
    with open(log_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 过滤掉警告信息
    lines = content.split('\n')
    filtered_lines = [line for line in lines if 'RuntimeWarning' not in line]
    filtered_content = '\n'.join(filtered_lines)

    # 使用字典来合并同一任务的数据
    task_data = defaultdict(dict)

    # 按完整的任务块分割（包括相似度和L2距离部分）
    # 使用更精确的分割方式，确保捕获完整的任务记录
    # 增强任务块分割逻辑，支持更复杂的任务名称
    task_blocks = re.split(r'(?=任务: \w+(?:-\w+)*)', filtered_content)
    task_blocks = [block.strip() for block in task_blocks if block.strip()]

    # 添加调试信息
    print(f"识别到 {len(task_blocks)} 个任务块")
    for i, block in enumerate(task_blocks):
        first_line = block.split('\n')[0] if block else "空块"
        print(f"任务块 {i + 1}: {first_line[:100]}...")

    for block in task_blocks:
        lines = block.split('\n')
        record_info = {}

        # 提取任务基本信息
        for line in lines:
            # 提取任务名称
            if line.startswith('任务: '):
                # 支持包含连字符的任务名称
                task_match = re.match(r'任务: (\w+(?:-\w+)*)', line)
                if task_match:
                    record_info['Tasks'] = task_match.group(1)

            # 提取受害者模型
            elif '受害者模型:' in line:
                victim_match = re.search(r'受害者模型: ([A-Za-z0-9]+) \(文件缩写: ([A-Za-z0-9]+)\)', line)
                if victim_match:
                    record_info['Victim'] = victim_match.group(2)

            # 提取方法
            elif '方法:' in line:
                method_match = re.search(r'方法: ([A-Za-z]+)', line)
                if method_match:
                    record_info['Attack'] = method_match.group(1)

            # 提取输入文件并从中提取Target
            elif '输入文件:' in line:
                file_match = re.search(r'输入文件: (.*\.csv)', line)
                if file_match:
                    input_file = file_match.group(1)
                    parts = input_file.split('/')
                    for i, part in enumerate(parts):
                        if part in ['Authorship_Attribution', 'Clone_detection', 'Defect_detection'] and i + 1 < len(
                                parts):
                            record_info['Target'] = parts[i + 1]
                            break
                    else:
                        record_info['Target'] = 'Unknown'

        # 创建任务唯一标识
        if all(key in record_info for key in ['Tasks', 'Target', 'Attack', 'Victim']):
            task_key = f"{record_info['Tasks']}_{record_info['Target']}_{record_info['Attack']}_{record_info['Victim']}"

            # 如果这个任务已经存在，则更新记录，否则创建新记录
            if task_key not in task_data:
                # 初始化所有字段
                task_data[task_key] = {
                    'Tasks': record_info.get('Tasks', 'Unknown'),
                    'Target': record_info.get('Target', 'Unknown'),
                    'Attack': record_info.get('Attack', 'Unknown'),
                    'Victim': record_info.get('Victim', 'Unknown'),
                    'succ': 'nan',
                    'fail': 'nan',
                    'COS_transfer_success_avg': 'nan', 'COS_transfer_success_min': 'nan',
                    'COS_transfer_success_max': 'nan',
                    'COS_transfer_failure_avg': 'nan', 'COS_transfer_failure_min': 'nan',
                    'COS_transfer_failure_max': 'nan',
                    'Feature_COS_transfer_success_avg': 'nan', 'Feature_COS_transfer_success_min': 'nan',
                    'Feature_COS_transfer_success_max': 'nan',
                    'Feature_COS_transfer_failure_avg': 'nan', 'Feature_COS_transfer_failure_min': 'nan',
                    'Feature_COS_transfer_failure_max': 'nan',
                    'L2_transfer_success_avg': 'nan', 'L2_transfer_success_min': 'nan',
                    'L2_transfer_success_max': 'nan',
                    'L2_transfer_failure_avg': 'nan', 'L2_transfer_failure_min': 'nan',
                    'L2_transfer_failure_max': 'nan',
                    'Feature_L2_transfer_success_avg': 'nan', 'Feature_L2_transfer_success_min': 'nan',
                    'Feature_L2_transfer_success_max': 'nan',
                    'Feature_L2_transfer_failure_avg': 'nan', 'Feature_L2_transfer_failure_min': 'nan',
                    'Feature_L2_transfer_failure_max': 'nan'
                }

            current_record = task_data[task_key]

            # 解析统计信息
            current_section = None
            for line in lines:
                line = line.strip()

                # 确定当前部分
                if '迁移成功组' in line and '相似度' in line:
                    current_section = 'success_cos'
                elif '迁移失败组' in line and '相似度' in line:
                    current_section = 'failure_cos'
                elif '迁移成功组' in line and 'L2距离' in line:
                    current_section = 'success_l2'
                elif '迁移失败组' in line and 'L2距离' in line:
                    current_section = 'failure_l2'
                elif '迁移成功样本:' in line:
                    current_section = 'feature_success'
                elif '迁移失败样本:' in line:
                    current_section = 'feature_failure'
                elif line.startswith('任务: ') or '受害者模型:' in line or '方法:' in line or '输入文件:' in line:
                    # 跳过基本信息行
                    continue

                # 解析样本数量
                if '样本数量:' in line:
                    count_match = re.search(r'样本数量: (\d+)', line)
                    if count_match and current_section:
                        count = count_match.group(1)
                        if 'success' in current_section and 'feature' not in current_section:
                            current_record['succ'] = count
                        elif 'failure' in current_section and 'feature' not in current_section:
                            current_record['fail'] = count

                # 解析相似度信息
                elif '平均相似度:' in line and '特征' not in line:
                    avg_match = re.search(r'平均相似度: ([\d.]+)', line)
                    if avg_match and current_section in ['success_cos', 'failure_cos']:
                        avg = avg_match.group(1)
                        if current_section == 'success_cos':
                            current_record['COS_transfer_success_avg'] = avg
                        elif current_section == 'failure_cos':
                            current_record['COS_transfer_failure_avg'] = avg

                elif '最小相似度:' in line and '特征' not in line:
                    min_match = re.search(r'最小相似度: ([\d.]+)', line)
                    if min_match and current_section in ['success_cos', 'failure_cos']:
                        min_val = min_match.group(1)
                        if current_section == 'success_cos':
                            current_record['COS_transfer_success_min'] = min_val
                        elif current_section == 'failure_cos':
                            current_record['COS_transfer_failure_min'] = min_val

                elif '最大相似度:' in line and '特征' not in line:
                    max_match = re.search(r'最大相似度: ([\d.]+)', line)
                    if max_match and current_section in ['success_cos', 'failure_cos']:
                        max_val = max_match.group(1)
                        if current_section == 'success_cos':
                            current_record['COS_transfer_success_max'] = max_val
                        elif current_section == 'failure_cos':
                            current_record['COS_transfer_failure_max'] = max_val

                # 解析L2距离信息
                elif '平均距离:' in line and '特征' not in line:
                    avg_match = re.search(r'平均距离: ([\d.]+)', line)
                    if avg_match and current_section in ['success_l2', 'failure_l2']:
                        avg = avg_match.group(1)
                        if current_section == 'success_l2':
                            current_record['L2_transfer_success_avg'] = avg
                        elif current_section == 'failure_l2':
                            current_record['L2_transfer_failure_avg'] = avg

                elif '最小距离:' in line and '特征' not in line:
                    min_match = re.search(r'最小距离: ([\d.]+)', line)
                    if min_match and current_section in ['success_l2', 'failure_l2']:
                        min_val = min_match.group(1)
                        if current_section == 'success_l2':
                            current_record['L2_transfer_success_min'] = min_val
                        elif current_section == 'failure_l2':
                            current_record['L2_transfer_failure_min'] = min_val

                elif '最大距离:' in line and '特征' not in line:
                    max_match = re.search(r'最大距离: ([\d.]+)', line)
                    if max_match and current_section in ['success_l2', 'failure_l2']:
                        max_val = max_match.group(1)
                        if current_section == 'success_l2':
                            current_record['L2_transfer_success_max'] = max_val
                        elif current_section == 'failure_l2':
                            current_record['L2_transfer_failure_max'] = max_val

                # 解析特征相似度信息
                elif '特征平均相似度:' in line:
                    avg_match = re.search(r'特征平均相似度: ([\d.-]+)', line)
                    if avg_match and current_section in ['feature_success', 'feature_failure']:
                        avg = avg_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_COS_transfer_success_avg'] = avg
                        elif current_section == 'feature_failure':
                            current_record['Feature_COS_transfer_failure_avg'] = avg

                elif '特征最小相似度:' in line:
                    min_match = re.search(r'特征最小相似度: ([\d.-]+)', line)
                    if min_match and current_section in ['feature_success', 'feature_failure']:
                        min_val = min_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_COS_transfer_success_min'] = min_val
                        elif current_section == 'feature_failure':
                            current_record['Feature_COS_transfer_failure_min'] = min_val

                elif '特征最大相似度:' in line:
                    max_match = re.search(r'特征最大相似度: ([\d.-]+)', line)
                    if max_match and current_section in ['feature_success', 'feature_failure']:
                        max_val = max_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_COS_transfer_success_max'] = max_val
                        elif current_section == 'feature_failure':
                            current_record['Feature_COS_transfer_failure_max'] = max_val

                # 解析特征L2距离信息
                elif '特征平均L2距离:' in line:
                    avg_match = re.search(r'特征平均L2距离: ([\d.]+)', line)
                    if avg_match and current_section in ['feature_success', 'feature_failure']:
                        avg = avg_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_L2_transfer_success_avg'] = avg
                        elif current_section == 'feature_failure':
                            current_record['Feature_L2_transfer_failure_avg'] = avg

                elif '特征最小L2距离:' in line:
                    min_match = re.search(r'特征最小L2距离: ([\d.]+)', line)
                    if min_match and current_section in ['feature_success', 'feature_failure']:
                        min_val = min_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_L2_transfer_success_min'] = min_val
                        elif current_section == 'feature_failure':
                            current_record['Feature_L2_transfer_failure_min'] = min_val

                elif '特征最大L2距离:' in line:
                    max_match = re.search(r'特征最大L2距离: ([\d.]+)', line)
                    if max_match and current_section in ['feature_success', 'feature_failure']:
                        max_val = max_match.group(1)
                        if current_section == 'feature_success':
                            current_record['Feature_L2_transfer_success_max'] = max_val
                        elif current_section == 'feature_failure':
                            current_record['Feature_L2_transfer_failure_max'] = max_val

    # 将字典转换为列表
    return list(task_data.values())


def write_to_csv(data, output_file):
    """将数据写入CSV文件"""
    if not data:
        print("No data to write.")
        return

    # 按照您提供的字段顺序
    fieldnames = [
        'Tasks', 'Target', 'Attack', 'Victim', 'succ', 'fail',
        'COS_transfer_success_avg', 'COS_transfer_success_min', 'COS_transfer_success_max',
        'COS_transfer_failure_avg', 'COS_transfer_failure_min', 'COS_transfer_failure_max',
        'Feature_COS_transfer_success_avg', 'Feature_COS_transfer_success_min', 'Feature_COS_transfer_success_max',
        'Feature_COS_transfer_failure_avg', 'Feature_COS_transfer_failure_min', 'Feature_COS_transfer_failure_max',
        'L2_transfer_success_avg', 'L2_transfer_success_min', 'L2_transfer_success_max',
        'L2_transfer_failure_avg', 'L2_transfer_failure_min', 'L2_transfer_failure_max',
        'Feature_L2_transfer_success_avg', 'Feature_L2_transfer_success_min', 'Feature_L2_transfer_success_max',
        'Feature_L2_transfer_failure_avg', 'Feature_L2_transfer_failure_min', 'Feature_L2_transfer_failure_max'
    ]

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in data:
                # 确保记录包含所有字段
                for field in fieldnames:
                    if field not in record:
                        record[field] = 'nan'
                writer.writerow(record)
        print(f"数据已成功提取到 {output_file}")
    except Exception as e:
        print(f"写入CSV文件时出错: {e}")


# 主程序
if __name__ == "__main__":
    log_file_path = "analysis.log"  # 替换为你的日志文件路径
    output_csv = "extracted_data.csv"

    if not os.path.exists(log_file_path):
        print(f"日志文件 {log_file_path} 不存在。")
    else:
        data = parse_log(log_file_path)
        if data:
            print(f"成功解析 {len(data)} 条任务记录")
            for i, record in enumerate(data, 1):
                print(
                    f"记录 {i}: {record['Tasks']} - {record['Target']} - {record['Attack']} - {record['Victim']} - 成功样本: {record['succ']}, 失败样本: {record['fail']}")
        else:
            print("未解析到任何数据，请检查日志文件格式。")
        write_to_csv(data, output_csv)
