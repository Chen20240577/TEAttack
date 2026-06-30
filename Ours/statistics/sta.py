import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签，SimHei为黑体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# 创建结果目录
if not os.path.exists('correlation_results'):
    os.makedirs('correlation_results')

# 读取数据
df = pd.read_excel('扰动.xlsx')

# 数据预处理 - 将分类变量编码为数值
le_target = LabelEncoder()
le_attack = LabelEncoder()
le_victim = LabelEncoder()

df['Target_encoded'] = le_target.fit_transform(df['Target'])
df['Attack_encoded'] = le_attack.fit_transform(df['Attack'])
df['Victim_encoded'] = le_victim.fit_transform(df['Victim'])
df['TY_encoded'] = le_victim.fit_transform(df['TY'])
df['TSR_encoded'] = le_victim.fit_transform(df['TSR'])

# 定义分类变量和数值变量
# categorical_cols = ['Target_encoded', 'Attack_encoded', 'Victim_encoded', 'TY_encoded', 'TSR_encoded']
# categorical_names = ['Target', 'Attack', 'Victim', 'TY', 'TSR']  # 用于显示的列名
categorical_cols = ['TY_encoded', 'TSR_encoded']
categorical_names = ['TY', 'TSR']  # 用于显示的列名

numeric_columns = ['succ_COS', 'fail_COS', 'succ_F_COS', 'fail_F_COS',
                   'succ_L2', 'fail_L2', 'succ_F_L2', 'fail_F_L2']

# 创建一个空的DataFrame来存储所有任务的相关性结果
all_correlations_summary = pd.DataFrame()

print("=" * 80)

# 按Tasks分组处理
tasks = df['Tasks'].unique()

for task in tasks:
    print(f"\n{'=' * 40} {task} {'=' * 40}")
    task_data = df[df['Tasks'] == task].copy()

    if len(task_data) == 0:
        print(f"Task {task} 没有数据")
        continue

    print(f"任务 {task} 的数据量: {len(task_data)}")

    # 1. 描述性统计
    print(f"\n1. {task} - 数值变量的描述性统计:")
    print(task_data[numeric_columns].describe())

    # 2. 计算分类变量与数值变量的相关性矩阵
    # 创建一个只包含分类变量和数值变量的数据子集
    analysis_data = task_data[categorical_cols + numeric_columns].copy()

    # 计算Pearson相关性
    print(f"\n2. {task} - 分类变量与数值变量的Pearson相关性分析:")
    pearson_corr = analysis_data.corr(method='pearson')

    # 只提取分类变量与数值变量之间的相关性
    cat_num_corr = pearson_corr.loc[categorical_cols, numeric_columns]

    # 重命名索引以便更好理解
    cat_num_corr.index = categorical_names

    print("\n分类变量与数值变量的Pearson相关系数:")
    print(cat_num_corr.round(4))

    # 计算Spearman相关性
    print(f"\n3. {task} - 分类变量与数值变量的Spearman相关性分析:")
    spearman_corr = analysis_data.corr(method='spearman')

    # 只提取分类变量与数值变量之间的相关性
    cat_num_spearman = spearman_corr.loc[categorical_cols, numeric_columns]

    # 重命名索引以便更好理解
    cat_num_spearman.index = categorical_names

    print("\n分类变量与数值变量的Spearman相关系数:")
    print(cat_num_spearman.round(4))

    # 3. 可视化 - 只显示分类变量与数值变量的相关性热图
    plt.figure(figsize=(14, 6))

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pearson相关性热图
    sns.heatmap(cat_num_corr,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.3f',
                square=False,
                cbar_kws={"shrink": .8},
                ax=ax1)
    ax1.set_title(f'{task} - Pearson相关性\n(分类变量 vs 数值变量)', fontsize=12, fontweight='bold')

    # Spearman相关性热图
    sns.heatmap(cat_num_spearman,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.3f',
                square=False,
                cbar_kws={"shrink": .8},
                ax=ax2)
    ax2.set_title(f'{task} - Spearman相关性\n(分类变量 vs 数值变量)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'correlation_results/{task}_cat_num_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{task} 的分类变量与数值变量相关性热图已保存为: correlation_results/{task}_cat_num_correlation.png")

    # 4. 保存完整的相关性结果到表格
    # 为每个任务创建完整的结果表格
    pearson_flat = cat_num_corr.stack().reset_index()
    pearson_flat.columns = ['分类变量', '数值变量', 'Pearson相关系数']
    pearson_flat['任务'] = task
    pearson_flat['Pearson绝对值'] = pearson_flat['Pearson相关系数'].abs()

    spearman_flat = cat_num_spearman.stack().reset_index()
    spearman_flat.columns = ['分类变量', '数值变量', 'Spearman相关系数']
    spearman_flat['任务'] = task
    spearman_flat['Spearman绝对值'] = spearman_flat['Spearman相关系数'].abs()

    # 合并Pearson和Spearman结果 - 修复合并问题
    task_correlations = pd.merge(pearson_flat, spearman_flat,
                                 on=['任务', '分类变量', '数值变量'],
                                 how='outer')

    # 重新排列列的顺序
    task_correlations = task_correlations[['任务', '分类变量', '数值变量',
                                           'Pearson相关系数', 'Pearson绝对值',
                                           'Spearman相关系数', 'Spearman绝对值']]

    # 保存每个任务的完整相关性结果
    task_correlations.to_excel(f'correlation_results/{task}_correlations.xlsx', index=False)
    print(f"{task} 的完整相关性结果已保存为: correlation_results/{task}_correlations.xlsx")
    print(f"任务 {task} 的相关性结果形状: {task_correlations.shape}")

    # 添加到总的结果中
    all_correlations_summary = pd.concat([all_correlations_summary, task_correlations], ignore_index=True)

    # 5. 找出最强的相关性
    print(f"\n4. {task} - 最强的相关性关系:")

    # 处理Pearson相关性
    top_positive_pearson = pearson_flat.nlargest(10, 'Pearson相关系数')
    top_negative_pearson = pearson_flat.nlargest(10, 'Pearson绝对值')
    top_negative_pearson = top_negative_pearson[top_negative_pearson['Pearson相关系数'] < 0]  # 只保留负相关

    print("\n最强的正相关 (Pearson):")
    for _, row in top_positive_pearson.iterrows():
        print(f"  {row['分类变量']} 与 {row['数值变量']}: {row['Pearson相关系数']:.4f}")

    if len(top_negative_pearson) > 0:
        print("\n最强的负相关 (Pearson):")
        for _, row in top_negative_pearson.iterrows():
            print(f"  {row['分类变量']} 与 {row['数值变量']}: {row['Pearson相关系数']:.4f}")

    # 处理Spearman相关性
    top_positive_spearman = spearman_flat.nlargest(10, 'Spearman相关系数')
    top_negative_spearman = spearman_flat.nlargest(10, 'Spearman绝对值')
    top_negative_spearman = top_negative_spearman[top_negative_spearman['Spearman相关系数'] < 0]  # 只保留负相关

    print("\n最强的正相关 (Spearman):")
    for _, row in top_positive_spearman.iterrows():
        print(f"  {row['分类变量']} 与 {row['数值变量']}: {row['Spearman相关系数']:.4f}")

    if len(top_negative_spearman) > 0:
        print("\n最强的负相关 (Spearman):")
        for _, row in top_negative_spearman.iterrows():
            print(f"  {row['分类变量']} 与 {row['数值变量']}: {row['Spearman相关系数']:.4f}")

# 保存所有任务的汇总结果
print("\n" + "=" * 80)
print("正在保存所有任务的汇总相关性结果...")

# 保存合并的相关性结果
if not all_correlations_summary.empty:
    all_correlations_summary.to_excel('correlation_results/all_tasks_correlations_summary.xlsx', index=False)
    print("所有任务的合并相关性结果已保存为: correlation_results/all_tasks_correlations_summary.xlsx")
    print(f"汇总结果形状: {all_correlations_summary.shape}")
else:
    print("警告: 汇总结果为空，将尝试重新生成...")

    # 重新生成汇总结果
    all_correlations_summary = pd.DataFrame()
    for task in tasks:
        try:
            task_corr = pd.read_excel(f'correlation_results/{task}_correlations.xlsx')
            all_correlations_summary = pd.concat([all_correlations_summary, task_corr], ignore_index=True)
        except FileNotFoundError:
            print(f"文件 {task}_correlations.xlsx 不存在，跳过")

    if not all_correlations_summary.empty:
        all_correlations_summary.to_excel('correlation_results/all_tasks_correlations_summary.xlsx', index=False)
        print("重新生成的汇总结果已保存为: correlation_results/all_tasks_correlations_summary.xlsx")
        print(f"汇总结果形状: {all_correlations_summary.shape}")
    else:
        print("错误: 无法重新生成汇总结果")

# 创建单独保存的Pearson和Spearman结果
print("\n生成单独的相关性结果文件...")
all_pearson = pd.DataFrame()
all_spearman = pd.DataFrame()

for task in tasks:
    task_data = df[df['Tasks'] == task].copy()
    if len(task_data) == 0:
        continue

    analysis_data = task_data[categorical_cols + numeric_columns].copy()

    # Pearson相关性
    pearson_corr = analysis_data.corr(method='pearson')
    cat_num_pearson = pearson_corr.loc[categorical_cols, numeric_columns]
    cat_num_pearson.index = categorical_names

    # Spearman相关性
    spearman_corr = analysis_data.corr(method='spearman')
    cat_num_spearman = spearman_corr.loc[categorical_cols, numeric_columns]
    cat_num_spearman.index = categorical_names

    # 转换为长格式
    pearson_flat = cat_num_pearson.stack().reset_index()
    pearson_flat.columns = ['分类变量', '数值变量', '相关系数']
    pearson_flat['方法'] = 'Pearson'
    pearson_flat['任务'] = task

    spearman_flat = cat_num_spearman.stack().reset_index()
    spearman_flat.columns = ['分类变量', '数值变量', '相关系数']
    spearman_flat['方法'] = 'Spearman'
    spearman_flat['任务'] = task

    # 合并到总结果
    all_pearson = pd.concat([all_pearson, pearson_flat], ignore_index=True)
    all_spearman = pd.concat([all_spearman, spearman_flat], ignore_index=True)

# 保存单独的结果
if not all_pearson.empty:
    all_pearson.to_excel('correlation_results/all_tasks_pearson_correlations.xlsx', index=False)
    print("所有任务的Pearson相关性结果已保存为: correlation_results/all_tasks_pearson_correlations.xlsx")

if not all_spearman.empty:
    all_spearman.to_excel('correlation_results/all_tasks_spearman_correlations.xlsx', index=False)
    print("所有任务的Spearman相关性结果已保存为: correlation_results/all_tasks_spearman_correlations.xlsx")

# 创建合并的详细结果
all_correlations_detailed = pd.concat([all_pearson, all_spearman], ignore_index=True)
if not all_correlations_detailed.empty:
    all_correlations_detailed.to_excel('correlation_results/all_tasks_correlations_detailed.xlsx', index=False)
    print("所有任务的详细相关性结果已保存为: correlation_results/all_tasks_correlations_detailed.xlsx")

# 跨任务比较分析
print("\n跨任务比较分析:")
print("=" * 50)

# 比较不同任务的TSR
task_tsr_comparison = df.groupby('Tasks')['TSR'].agg(['mean', 'std', 'count'])
print("各任务的TSR比较:")
print(task_tsr_comparison.round(4))

# 比较不同任务的攻击成功率
if 'succ' in df.columns and 'AEs_valid' in df.columns:
    df['success_rate'] = df['succ'] / df['AEs_valid']
    task_success_comparison = df.groupby('Tasks')['success_rate'].agg(['mean', 'std'])
    print("\n各任务的攻击成功率比较:")
    print(task_success_comparison.round(4))

print("\n所有分析完成！")
print("=" * 80)
print("生成的文件列表:")
print("1. 每个任务的相关性热图: correlation_results/[task]_cat_num_correlation.png")
print("2. 每个任务的完整相关性表格: correlation_results/[task]_correlations.xlsx")
print("3. 所有任务的合并相关性汇总: correlation_results/all_tasks_correlations_summary.xlsx")
print("4. 所有任务的Pearson相关性汇总: correlation_results/all_tasks_pearson_correlations.xlsx")
print("5. 所有任务的Spearman相关性汇总: correlation_results/all_tasks_spearman_correlations.xlsx")
print("6. 所有任务的详细相关性结果: correlation_results/all_tasks_correlations_detailed.xlsx")
