import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 读取Excel文件并处理数据
file_path = "disturb0.xls"
sheet_name = "Sheet1"

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    column_names = df.iloc[0].tolist()
    data = df.iloc[1:].reset_index(drop=True)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.columns = column_names
    print(f"Data loaded successfully! Shape: {data.shape}")

except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 2. 创建2x2排列的4个子图
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# 3. 定义4个图表的数据对
chart_pairs = [
    (0, 1, '#F4704F', '#6AADD7'),  # 第1对
    (2, 3, '#F4704F', '#6AADD7'),  # 第2对
    (4, 5, '#F4704F', '#6AADD7'),  # 第3对
    (6, 7, '#F4704F', '#6AADD7')   # 第4对
]

# 4. 绘制4个图表
for idx, (col1_idx, col2_idx, color1, color2) in enumerate(chart_pairs):
    if col1_idx < len(data.columns) and col2_idx < len(data.columns):
        col1_name = data.columns[col1_idx]
        col2_name = data.columns[col2_idx]

        ax = axes.flat[idx]

        # 绘制折线
        line1 = ax.plot(data.index, data.iloc[:, col1_idx],
                        color=color1, linewidth=2, label="succ")[0]
        line2 = ax.plot(data.index, data.iloc[:, col2_idx],
                        color=color2, linewidth=2, label="fail")[0]

        # 设置子图属性
        ax.set_xlabel(col1_name, fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.6)

        # 只在第二个子图显示图例
        if idx == 1:  # 第二个子图（索引为1）
            ax.legend(loc='best', fontsize=25)

        # Y轴范围调整
        y_min = min(data.iloc[:, col1_idx].min(), data.iloc[:, col2_idx].min())
        y_max = max(data.iloc[:, col1_idx].max(), data.iloc[:, col2_idx].max())

        margin = (y_max - y_min) * 0.1
        if margin < 0.001:
            margin = 0.001

        if y_min >= 0:
            y_min_adj = max(0, y_min - margin)
        else:
            y_min_adj = y_min - margin

        ax.set_ylim([y_min_adj, y_max + margin])

        # X轴刻度
        tick_indices = np.arange(0, len(data), max(1, len(data) // 10))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(int(i)) for i in tick_indices], fontsize=8)

plt.tight_layout()

# 5. 显示图表
print("\nShowing 2x2 charts...")
plt.show()

# 6. 保存图表
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "four_dimensions_origin.pdf")
fig.savefig(save_path, bbox_inches='tight', facecolor='white')
print(f"\nCharts saved to: {save_path}")

# 7. 打印Y轴范围信息
print("\nY-axis ranges for each subplot:")
for idx, (col1_idx, col2_idx, color1, color2) in enumerate(chart_pairs):
    if col1_idx < len(data.columns) and col2_idx < len(data.columns):
        col1_name = data.columns[col1_idx]
        y_min = min(data.iloc[:, col1_idx].min(), data.iloc[:, col2_idx].min())
        y_max = max(data.iloc[:, col1_idx].max(), data.iloc[:, col2_idx].max())
        print(f"{col1_name}: {y_min:.6f} to {y_max:.6f} (range: {y_max - y_min:.6f})")