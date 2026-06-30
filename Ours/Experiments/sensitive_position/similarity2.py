import ast
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set font for better visualization
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 18

# Create output directory
OUTPUT_DIR = "./image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration constants
CONFIG = {
    'required_score_columns': 4,
    'min_valid_rows': 1,
    'similarity_range': (0, 1)
}


def safe_literal_eval(dict_str: Union[str, Dict]) -> Optional[Dict]:
    """
    Safely evaluate string representation of dictionary

    Args:
        dict_str: String representation of dictionary or dict object

    Returns:
        Optional[Dict]: Parsed dictionary or None
    """
    if isinstance(dict_str, dict):
        return dict_str

    if not isinstance(dict_str, str):
        logger.warning(f"Expected string or dict, got {type(dict_str)}")
        return None

    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"Parsed result is not a dictionary: {type(result)}")
            return None

    except (SyntaxError, ValueError) as e:
        logger.debug(f"Dictionary parsing syntax error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unknown error in dictionary parsing: {e}")
        return None


def validate_and_filter_dataframe(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """
    Validate DataFrame and filter rows with null values in specified columns

    Args:
        df: Original DataFrame
        score_columns: List of column names to check

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    original_rows = len(df)

    # Check null values
    null_counts = df[score_columns].isnull().sum()
    total_nulls = null_counts.sum()

    logger.info(f"Null counts per column: {null_counts.to_dict()}")
    logger.info(f"Total null values: {total_nulls}")

    # Filter rows where any of the four Score columns is null
    df_filtered = df.dropna(subset=score_columns)
    filtered_rows = len(df_filtered)

    logger.info(f"Original rows: {original_rows}")
    logger.info(f"Valid rows after filtering (all four columns non-null): {filtered_rows}")

    if filtered_rows < original_rows:
        filtered_percentage = (1 - filtered_rows / original_rows) * 100
        logger.info(f"Null value filtering rate: {filtered_percentage:.2f}%")

    return df_filtered


def process_single_csv_file(csv_path: str, task_name: str) -> Tuple[
    Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[str]]]:
    """
    Process single CSV file and return data for visualization

    Args:
        csv_path: Path to CSV file
        task_name: Task name

    Returns:
        Tuple containing:
        - Average similarity matrix
        - All similarity matrices
        - Model names
    """
    logger.info(f"Processing task: {task_name}")

    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return None, None, None

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully read file: {csv_path}, total rows: {len(df)}")
    except Exception as e:
        logger.error(f"Error reading file {csv_path}: {e}")
        return None, None, None

    # Find columns containing "Score"
    score_columns = [col for col in df.columns if 'Score' in col]
    if len(score_columns) < CONFIG['required_score_columns']:
        logger.error(f"Insufficient Score columns in {csv_path}, found: {score_columns}")
        return None, None, None

    # Use first 4 Score columns
    model_columns = score_columns[:CONFIG['required_score_columns']]
    model_names = [col.replace('Score', '').strip() for col in model_columns]
    logger.info(f"Models used: {model_names}")

    # Filter rows with null values in any of the four Score columns
    df_filtered = validate_and_filter_dataframe(df, model_columns)

    if len(df_filtered) == 0:
        logger.error(f"No valid data (all four columns non-null) for task {task_name}")
        return None, None, None

    # Store similarity matrices for all rows
    all_similarities = []
    valid_rows = 0
    parse_errors = 0

    for idx, row in df_filtered.iterrows():
        try:
            dicts = []
            valid_row = True

            # Parse dictionaries from four columns
            for col in model_columns:
                score_dict = safe_literal_eval(row[col])
                if score_dict is None:
                    parse_errors += 1
                    valid_row = False
                    break
                dicts.append(score_dict)

            if not valid_row:
                continue

            # Collect all unique keys
            all_keys = []
            seen = set()
            for d in dicts:
                for k in d.keys():
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)

            # Vectorize four models based on key order
            vectors = []
            for d in dicts:
                vec = [float(d.get(k, 0.0)) for k in all_keys]  # missing key → 0
                vectors.append(vec)

            # Calculate 4×4 similarity matrix
            row_similarities = np.zeros((CONFIG['required_score_columns'],
                                         CONFIG['required_score_columns']))

            for i in range(CONFIG['required_score_columns']):
                for j in range(CONFIG['required_score_columns']):
                    v1 = np.array(vectors[i], dtype=float).reshape(1, -1)
                    v2 = np.array(vectors[j], dtype=float).reshape(1, -1)
                    sim = cosine_similarity(v1, v2)[0][0]

                    # Force self-similarity to 1
                    if i == j:
                        sim = 1.0

                    row_similarities[i][j] = sim

            all_similarities.append(row_similarities)
            valid_rows += 1

        except Exception as e:
            parse_errors += 1
            logger.debug(f"Error processing row {idx}: {e}")
            continue

    if parse_errors > 0:
        logger.warning(f"Dictionary parsing failed rows: {parse_errors}")

    logger.info(f"Task {task_name} final valid rows: {valid_rows}")

    if valid_rows < CONFIG['min_valid_rows']:
        logger.error(f"Task {task_name} insufficient valid data, need at least {CONFIG['min_valid_rows']} rows")
        return None, None, None

    # Calculate average similarity matrix
    avg_similarity_matrix = np.mean(all_similarities, axis=0)

    return avg_similarity_matrix, all_similarities, model_names


def generate_statistics_report(avg_matrix: np.ndarray, boxplot_data: List[List[float]],
                               unique_pairs: List[str], task_name: str, num_samples: int,
                               original_rows: int, filtered_rows: int) -> None:
    """
    Generate detailed statistics report in English

    Args:
        avg_matrix: Average similarity matrix
        boxplot_data: Boxplot data
        unique_pairs: Model pair names
        task_name: Task name
        num_samples: Number of samples
        original_rows: Original row count
        filtered_rows: Filtered row count
    """
    report_path = os.path.join(OUTPUT_DIR, f"{task_name}_statistics.txt")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Statistical Analysis Report for {task_name} Task ===\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original rows: {original_rows}\n")
            f.write(f"Rows after filtering (all four columns non-null): {filtered_rows}\n")
            f.write(f"Null value filtering rate: {(1 - filtered_rows / original_rows) * 100:.2f}%\n")
            f.write(f"Final valid rows: {num_samples}\n")
            f.write(f"Data utilization rate: {(num_samples / original_rows) * 100:.2f}%\n\n")

            f.write("Average similarity matrix:\n")
            for i in range(avg_matrix.shape[0]):
                f.write("  " + " ".join([f"{val:8.4f}" for val in avg_matrix[i]]) + "\n")

            f.write("\nModel pair similarity statistics:\n")
            f.write("-" * 80 + "\n")

            for i, (pair, data) in enumerate(zip(unique_pairs, boxplot_data)):
                if data:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    median_val = np.median(data)
                    min_val = np.min(data)
                    max_val = np.max(data)

                    f.write(f"{pair:20}: ")
                    f.write(f"mean={mean_val:.4f}, std={std_val:.4f}, ")
                    f.write(f"median={median_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}], ")
                    f.write(f"samples={len(data)}\n")

            # Add overall statistics
            f.write("\nOverall statistics:\n")
            all_similarities = [item for sublist in boxplot_data for item in sublist]
            if all_similarities:
                f.write(f"Global mean similarity: {np.mean(all_similarities):.4f}\n")
                f.write(f"Global similarity standard deviation: {np.std(all_similarities):.4f}\n")

        logger.info(f"Statistics report saved to: {report_path}")

    except Exception as e:
        logger.error(f"Error generating statistics report: {e}")


def draw_combined_boxplots(task_data_list: List[dict]) -> None:
    """
    Draw combined boxplots for all tasks side by side with unified color scheme

    Args:
        task_data_list: List of dictionaries containing task data
    """
    try:
        n_tasks = len(task_data_list)

        # 修改图形尺寸：保持宽度不变，高度调整为原来的0.7倍
        # 原始高度为8，0.7倍为5.6
        fig_width = 7 * n_tasks
        fig_height = 6

        # Create figure with subplots (1 row, n_tasks columns)
        fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, fig_height),
                                 sharey=True, constrained_layout=True)

        # If only one task, make axes a list for consistent indexing
        if n_tasks == 1:
            axes = [axes]

        # Define a unified color palette for model pairs
        # We'll use the same 6 colors for all 6 model pairs (4 models = 6 pairs)
        unified_colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
        ]

        # First, determine the model pairs from the first task
        first_task = task_data_list[0]
        first_model_names = first_task['model_names']

        # Generate model pair names in a consistent order
        # This ensures the same color is used for the same model pair across all tasks
        model_pair_names = []
        for i in range(len(first_model_names)):
            for j in range(i + 1, len(first_model_names)):
                # Shorter display names for x-axis labels
                pair_name = f"{first_model_names[i][:3]}-{first_model_names[j][:3]}"
                model_pair_names.append(pair_name)

        # Create a mapping from model pair index to color
        color_mapping = {}
        for idx, pair_name in enumerate(model_pair_names):
            color_mapping[idx] = unified_colors[idx % len(unified_colors)]

        # Now draw each subplot
        for idx, (ax, task_data) in enumerate(zip(axes, task_data_list)):
            task_name = task_data['task_name']
            model_names = task_data['model_names']
            all_similarities = task_data['all_similarities']

            # Prepare data for boxplot
            unique_pairs = []
            boxplot_data = []

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    # Use shorter names for display
                    pair_name = f"{model_names[i][:3]}-{model_names[j][:3]}"
                    unique_pairs.append(pair_name)

                    # Extract all similarity values for this model pair
                    pair_similarities = []
                    for sim_matrix in all_similarities:
                        pair_similarities.append(sim_matrix[i][j])
                    boxplot_data.append(pair_similarities)

            # Create boxplot
            bp = ax.boxplot(boxplot_data, labels=unique_pairs, patch_artist=True,
                            showmeans=True, meanline=True, showfliers=True,
                            meanprops={'color': 'red', 'linewidth': 1.5},
                            medianprops={'color': 'black', 'linewidth': 2},
                            flierprops={'marker': 'o', 'markerfacecolor': 'gray',
                                        'markersize': 3, 'alpha': 0.6})

            # Apply consistent coloring based on model pair index
            for patch_idx, patch in enumerate(bp['boxes']):
                # Use the same color for the same model pair index
                color_idx = patch_idx % len(unified_colors)
                color = unified_colors[color_idx]
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)

            # Set title and labels
            ax.set_title(task_name, fontsize=30, fontweight='bold', pad=15)
            # ax.set_xlabel('Model Pairs', fontsize=16, fontweight='bold')
            if idx == 0:  # Only set ylabel for the first subplot
                ax.set_ylabel('Cosine Similarity', fontsize=25)

            # Rotate x-tick labels for better readability
            ax.tick_params(axis='x', rotation=20, labelsize=25)
            ax.tick_params(axis='y', labelsize=14)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # 移除具体数值标注
            # 注释掉下面这些行，不再显示均值和中位数的具体数值

        # Save figure
        output_path = os.path.join(OUTPUT_DIR, "combined_boxplots_unified_colors.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "combined_boxplots_unified_colors.pdf"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Combined boxplot with unified colors saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error drawing combined boxplots: {e}")
        raise


def main() -> None:
    """
    Main function to coordinate processing
    """
    # CSV file paths configuration
    csv_files = [
        ("./merged_record/authorship_merged.csv", "Authorship Attribution"),
        ("./merged_record/clone_merged.csv", "Clone Detection"),
        ("./merged_record/defect_merged.csv", "Defect Detection")
    ]

    logger.info("Starting CSV file processing...")
    logger.info("=" * 50)

    # Store task data for combined visualization
    task_data_list = []

    # Process each CSV file
    for csv_path, task_name in csv_files:
        if not os.path.exists(csv_path):
            logger.warning(f"File {csv_path} not found, skipping")
            continue

        logger.info(f"\nProcessing task: {task_name}")
        logger.info("-" * 30)

        result = process_single_csv_file(csv_path, task_name)
        if result[0] is not None and result[1] is not None and result[2] is not None:
            avg_matrix, all_similarities, model_names = result

            # Store data for combined visualization
            task_data = {
                'task_name': task_name,
                'avg_matrix': avg_matrix,
                'all_similarities': all_similarities,
                'model_names': model_names,
                'num_samples': len(all_similarities)
            }
            task_data_list.append(task_data)

            # Generate statistics report for this task
            if len(all_similarities) > 0:
                # Prepare boxplot data for statistics report
                unique_pairs = []
                boxplot_data = []

                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        pair_name = f"{model_names[i]}-{model_names[j]}"
                        unique_pairs.append(pair_name)

                        pair_similarities = []
                        for sim_matrix in all_similarities:
                            pair_similarities.append(sim_matrix[i][j])
                        boxplot_data.append(pair_similarities)

                # Generate statistics report
                generate_statistics_report(
                    avg_matrix, boxplot_data, unique_pairs, task_name,
                    len(all_similarities), 0, 0
                )

            logger.info(f"Task {task_name} processed successfully")

        logger.info("-" * 30)

    # Draw combined boxplots if we have data
    if task_data_list:
        draw_combined_boxplots(task_data_list)
        logger.info(f"Processing completed! Results saved to: {OUTPUT_DIR}")
    else:
        logger.error("No valid data processed. No output generated.")


if __name__ == "__main__":
    main()
    # 黑色实线（中位数）和红色虚线（均值）