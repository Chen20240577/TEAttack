import pandas as pd
from scipy.stats import pearsonr

# === 1. 自动读取 Excel ===
file_path = r"relate.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# === 2. 自动识别 TY / TSR ===
ty_col = "TY"
tsr_col = "TSR"
feature_cols = [c for c in df.columns if c not in (ty_col, tsr_col)]

# === 3. 计算 Pearson 相关系数 + p 值 ===
records = []

for col in feature_cols:
    r_ty, p_ty = pearsonr(df[col], df[ty_col])
    r_tsr, p_tsr = pearsonr(df[col], df[tsr_col])

    records.append({
        "Feature": col,
        "Pearson_TY": r_ty,
        "P_TY": p_ty,
        "Pearson_TSR": r_tsr,
        "P_TSR": p_tsr
    })

# === 4. 输出结果（按 TSR 排序）===
result = pd.DataFrame(records).sort_values("Pearson_TSR", ascending=False)
print(result)