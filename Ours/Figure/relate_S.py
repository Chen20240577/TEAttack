import pandas as pd
from scipy.stats import spearmanr

# === 1. 读数据（沿用你刚才的文件）===
file_path = r"relate.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# === 2. 自动识别特征列 ===
feature_cols = [c for c in df.columns if c not in ("TY", "TSR")]

# === 3. 计算 Spearman ===
records = []
for col in feature_cols:
    rho_ty, p_ty = spearmanr(df[col], df["TY"])
    rho_tsr, p_tsr = spearmanr(df[col], df["TSR"])
    records.append({
        "Feature": col,
        "Spearman_TY": rho_ty,
        "P_TY": p_ty,
        "Spearman_TSR": rho_tsr,
        "P_TSR": p_tsr
    })

result = pd.DataFrame(records)

# === 4. 按 TSR 排序 ===
result = result.sort_values("Spearman_TSR", ascending=False)
print(result)