import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = "./image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置常量
CONFIG = {
    'required_score_columns': 4,
    'min_valid_rows': 1,
    'similarity_range': (0, 1)
}


# def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
#     """
#     计算两个向量的余弦相似度
#
#     Args:
#         vec1: 第一个向量
#         vec2: 第二个向量
#
#     Returns:
#         float: 余弦相似度值，范围[0, 1]
#     """
#     try:
#         vec1_array = np.array(vec1, dtype=float)
#         vec2_array = np.array(vec2, dtype=float)
#
#         # 验证向量维度一致
#         if vec1_array.shape != vec2_array.shape:
#             logger.warning(f"向量维度不一致: {vec1_array.shape} vs {vec2_array.shape}")
#             return 0.0
#
#         dot_product = np.dot(vec1_array, vec2_array)
#         norm_vec1 = np.linalg.norm(vec1_array)
#         norm_vec2 = np.linalg.norm(vec2_array)
#
#         if norm_vec1 == 0 or norm_vec2 == 0:
#             return 0.0
#
#         similarity = dot_product / (norm_vec1 * norm_vec2)
#
#         # 确保相似度在合理范围内
#         return similarity
#
#     except Exception as e:
#         logger.error(f"计算余弦相似度时出错: {e}")
#         return 0.0


def safe_literal_eval(dict_str: Union[str, Dict]) -> Optional[Dict]:
    """
    安全地将字符串字典转换为字典对象

    Args:
        dict_str: 可能是字典字符串或字典对象

    Returns:
        Optional[Dict]: 解析后的字典或None
    """
    if isinstance(dict_str, dict):
        return dict_str

    if not isinstance(dict_str, str):
        logger.warning(f"预期字符串或字典，得到 {type(dict_str)}")
        return None

    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"解析结果不是字典: {type(result)}")
            return None

    except (SyntaxError, ValueError) as e:
        logger.debug(f"字典解析语法错误: {e}")
        return None
    except Exception as e:
        logger.error(f"字典解析未知错误: {e}")
        return None


def extract_vectors_from_dict(score_dict: Dict):
    """
    从字典中提取向量值，确保顺序一致并验证完整性

    Args:
        score_dict: 包含评分数据的字典
        expected_keys: 期望的键顺序（可选）

    Returns:
        Tuple[Optional[List], Optional[List]]: (向量值列表, 使用的键顺序)
    """
    if not isinstance(score_dict, dict) or len(score_dict) == 0:
        return None

    # 如果是第一次调用，建立期望的键顺序
    vector = [float(score_dict[key]) for key in score_dict.keys()]
    # vector = [abs(float(score_dict[key])) for key in score_dict.keys()]
    # 加不加绝对值得到的图一样
    return vector


def validate_and_filter_dataframe(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """
    验证DataFrame并过滤掉指定列中包含空值的行

    Args:
        df: 原始DataFrame
        score_columns: 需要检查的列名列表

    Returns:
        pd.DataFrame: 过滤后的DataFrame
    """
    original_rows = len(df)

    # 检查空值
    null_counts = df[score_columns].isnull().sum()
    total_nulls = null_counts.sum()

    logger.info(f"各列空值数量: {null_counts.to_dict()}")
    logger.info(f"总空值数量: {total_nulls}")

    # 🆕 关键修改：过滤掉四个Score列中任意一个为空值的行
    df_filtered = df.dropna(subset=score_columns)
    filtered_rows = len(df_filtered)

    logger.info(f"原始数据行数: {original_rows}")
    logger.info(f"过滤后有效行数（四列都不为空）: {filtered_rows}")

    if filtered_rows < original_rows:
        filtered_percentage = (1 - filtered_rows / original_rows) * 100
        logger.info(f"空值过滤率: {filtered_percentage:.2f}%")

    return df_filtered


def process_single_csv_file(csv_path: str, task_name: str) -> Optional[np.ndarray]:
    """
    处理单个CSV文件并生成可视化图表

    Args:
        csv_path: CSV文件路径
        task_name: 任务名称

    Returns:
        Optional[np.ndarray]: 平均相似度矩阵
    """
    logger.info(f"开始处理任务: {task_name}")

    if not os.path.exists(csv_path):
        logger.error(f"文件不存在: {csv_path}")
        return None

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取文件: {csv_path}, 共{len(df)}行数据")
    except Exception as e:
        logger.error(f"读取文件 {csv_path} 时出错: {e}")
        return None

    # 找出包含"Score"的列
    score_columns = [col for col in df.columns if 'Score' in col]
    if len(score_columns) < CONFIG['required_score_columns']:
        logger.error(f"文件 {csv_path} 中不足{CONFIG['required_score_columns']}个Score列，找到的列: {score_columns}")
        return None

    # 取前4个Score列
    model_columns = score_columns[:CONFIG['required_score_columns']]
    model_names = [col.replace('Score', '').strip() for col in model_columns]
    logger.info(f"使用的模型: {model_names}")

    # 🆕 关键修改：过滤掉四个Score列中任意一个为空值的行
    df_filtered = validate_and_filter_dataframe(df, model_columns)

    if len(df_filtered) == 0:
        logger.error(f"任务 {task_name} 没有四列都不为空的有效数据")
        return None

    # 存储所有行的相似度结果
    all_similarities = []
    valid_rows = 0
    parse_errors = 0
    original_rows = len(df)  # 保存原始行数用于报告

    for idx, row in df_filtered.iterrows():
        try:
            dicts = []
            valid_row = True

            # 解析四个列的字典
            for col in model_columns:
                score_dict = safe_literal_eval(row[col])
                if score_dict is None:
                    parse_errors += 1
                    valid_row = False
                    break
                dicts.append(score_dict)

            if not valid_row:
                continue

            # === 1. 收集这一行所有出现过的 key（保持一致顺序）===
            all_keys = []
            seen = set()
            for d in dicts:
                for k in d.keys():
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)

            # === 2. 根据 key 顺序，向量化四个模型 ===
            vectors = []
            for d in dicts:
                vec = [float(d.get(k, 0.0)) for k in all_keys]  # missing key → 0
                vectors.append(vec)

            # === 3. 计算 4×4 相似度矩阵 ===
            row_similarities = np.zeros((CONFIG['required_score_columns'],
                                         CONFIG['required_score_columns']))

            for i in range(CONFIG['required_score_columns']):
                for j in range(CONFIG['required_score_columns']):
                    v1 = np.array(vectors[i], dtype=float).reshape(1, -1)
                    v2 = np.array(vectors[j], dtype=float).reshape(1, -1)
                    sim = cosine_similarity(v1, v2)[0][0]

                    # 修正数值漂移，自己与自己的相似度强制为 1
                    if i == j:
                        sim = 1.0

                    row_similarities[i][j] = sim

            all_similarities.append(row_similarities)
            valid_rows += 1

        except Exception as e:
            parse_errors += 1
            logger.debug(f"处理第{idx}行时出错: {e}")
            continue

    # 汇总解析结果
    if parse_errors > 0:
        logger.warning(f"字典解析失败的行数: {parse_errors}")

    logger.info(f"任务 {task_name} 最终有效行数: {valid_rows}")

    if valid_rows < CONFIG['min_valid_rows']:
        logger.error(f"任务 {task_name} 有效数据不足，需要至少{CONFIG['min_valid_rows']}行")
        return None

    # 计算平均相似度矩阵
    avg_similarity_matrix = np.mean(all_similarities, axis=0)

    # 为当前任务生成图表
    generate_visualizations(avg_similarity_matrix, all_similarities,
                            model_names, task_name, valid_rows, original_rows, len(df_filtered))

    return avg_similarity_matrix


def generate_visualizations(avg_matrix: np.ndarray, all_similarities: List[np.ndarray],
                            model_names: List[str], task_name: str, num_samples: int,
                            original_rows: int, filtered_rows: int) -> None:
    """
    为单个任务生成热力图和箱型图

    Args:
        avg_matrix: 平均相似度矩阵
        all_similarities: 所有相似度数据
        model_names: 模型名称列表
        task_name: 任务名称
        num_samples: 样本数量
        original_rows: 原始数据行数
        filtered_rows: 过滤后行数
    """
    try:
        # 创建图形布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(
            f'任务分析: {task_name}\n(原始行数: {original_rows}, 过滤后: {filtered_rows}, 最终有效: {num_samples})',
            fontsize=16, fontweight='bold')

        # 1. 热力图
        im = ax1.imshow(avg_matrix, cmap='YlOrRd', aspect='auto',
                        vmin=CONFIG['similarity_range'][0],
                        vmax=CONFIG['similarity_range'][1])
        ax1.set_xticks(range(len(model_names)))
        ax1.set_yticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
        ax1.set_yticklabels(model_names, fontsize=12)
        ax1.set_title('模型间平均余弦相似度热力图', fontsize=14, fontweight='bold')

        # 添加数值标注
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                color = 'white' if avg_matrix[i, j] > 0.5 else 'black'
                ax1.text(j, i, f'{avg_matrix[i, j]:.3f}',
                         ha='center', va='center', fontsize=10, fontweight='bold', color=color)

        plt.colorbar(im, ax=ax1, shrink=0.8)

        # 2. 箱型图
        unique_pairs = []
        boxplot_data = []

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pair_name = f"{model_names[i]}-{model_names[j]}"
                unique_pairs.append(pair_name)

                # 提取该模型对的所有相似度值
                pair_similarities = []
                for sim_matrix in all_similarities:
                    pair_similarities.append(sim_matrix[i][j])
                boxplot_data.append(pair_similarities)

        # 绘制箱型图
        box_plot = ax2.boxplot(boxplot_data, labels=unique_pairs, patch_artist=True,
                               showmeans=True, meanline=True)
        ax2.set_title('模型对间余弦相似度分布箱型图', fontsize=14, fontweight='bold')
        ax2.set_ylabel('余弦相似度', fontsize=12)
        ax2.set_xlabel('模型对', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 设置箱型图颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_pairs)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(OUTPUT_DIR, f"{task_name}_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"任务 {task_name} 的可视化图表已保存至: {output_path}")

        # 生成详细统计信息
        generate_statistics_report(avg_matrix, boxplot_data, unique_pairs, task_name,
                                   num_samples, original_rows, filtered_rows)

    except Exception as e:
        logger.error(f"生成可视化图表时出错: {e}")
        raise


def generate_statistics_report(avg_matrix: np.ndarray, boxplot_data: List[List[float]],
                               unique_pairs: List[str], task_name: str, num_samples: int,
                               original_rows: int, filtered_rows: int) -> None:
    """
    生成详细的统计报告，包含空值处理信息

    Args:
        avg_matrix: 平均相似度矩阵
        boxplot_data: 箱型图数据
        unique_pairs: 模型对名称列表
        task_name: 任务名称
        num_samples: 样本数量
        original_rows: 原始数据行数
        filtered_rows: 过滤后行数
    """
    report_path = os.path.join(OUTPUT_DIR, f"{task_name}_statistics.txt")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {task_name} 任务统计分析报告 ===\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始数据行数: {original_rows}\n")
            f.write(f"四列都不为空的筛选后行数: {filtered_rows}\n")
            f.write(f"空值过滤率: {(1 - filtered_rows / original_rows) * 100:.2f}%\n")
            f.write(f"最终有效行数: {num_samples}\n")
            f.write(f"数据利用率: {(num_samples / original_rows) * 100:.2f}%\n\n")

            f.write("平均相似度矩阵:\n")
            for i in range(avg_matrix.shape[0]):
                f.write("  " + " ".join([f"{val:8.4f}" for val in avg_matrix[i]]) + "\n")

            f.write("\n模型对相似度统计:\n")
            f.write("-" * 80 + "\n")

            for i, (pair, data) in enumerate(zip(unique_pairs, boxplot_data)):
                if data:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    median_val = np.median(data)
                    min_val = np.min(data)
                    max_val = np.max(data)

                    f.write(f"{pair:20}: ")
                    f.write(f"均值={mean_val:.4f}, 标准差={std_val:.4f}, ")
                    f.write(f"中位数={median_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}], ")
                    f.write(f"样本数={len(data)}\n")

            # 添加总体统计
            f.write("\n总体统计:\n")
            all_similarities = [item for sublist in boxplot_data for item in sublist]
            if all_similarities:
                f.write(f"全局平均相似度: {np.mean(all_similarities):.4f}\n")
                f.write(f"全局相似度标准差: {np.std(all_similarities):.4f}\n")

        logger.info(f"统计报告已保存至: {report_path}")

    except Exception as e:
        logger.error(f"生成统计报告时出错: {e}")


def main() -> None:
    """
    主函数，协调整个处理流程
    """
    # CSV文件路径配置
    csv_files = [
        ("./merged_record/authorship_merged.csv", "authorship"),
        ("./merged_record/clone_merged.csv", "clone"),
        ("./merged_record/defect_merged.csv", "defect")
    ]

    logger.info("开始处理CSV文件...")
    logger.info("=" * 50)

    # 存储各任务结果
    task_results = {}

    # 处理每个CSV文件
    for csv_path, task_name in csv_files:
        if not os.path.exists(csv_path):
            logger.warning(f"文件 {csv_path} 不存在，跳过处理")
            continue

        logger.info(f"\n正在处理任务: {task_name}")
        logger.info("-" * 30)

        result = process_single_csv_file(csv_path, task_name)
        if result is not None:
            task_results[task_name] = result
        logger.info("-" * 30)

    logger.info(f"处理完成! 所有结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()