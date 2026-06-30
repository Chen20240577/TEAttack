# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from scipy.stats import (
    binomtest,
    wilcoxon,
    norm
)

# =====================================================
# Read Data
# =====================================================

file_path = r"relate.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# =====================================================
# Metrics Pairs
# =====================================================

pairs = [
    ("cos abs(succ-avg)", "cos abs(fail-avg)"),
    ("F_cos abs(succ-avg)", "F_cos abs(fail-avg)"),
    ("L2 abs(succ-avg)", "L2 abs(fail-avg)"),
    ("F_L2 abs(succ-avg)", "F_L2 abs(fail-avg)")
]

# =====================================================
# Analysis
# =====================================================

records = []

for succ_col, fail_col in pairs:

    print(f"\nProcessing: {succ_col}")

    # ---------------------------------
    # Difference
    # ---------------------------------

    diff = df[succ_col] - df[fail_col]

    # remove NaN
    diff = diff.dropna()

    # ---------------------------------
    # Win / Lose Count
    # ---------------------------------

    wins = (diff > 0).sum()

    losses = (diff < 0).sum()

    ties = (diff == 0).sum()

    total = wins + losses + ties

    win_rate = wins / total

    # ---------------------------------
    # Binomial Test
    # H1:
    # succ deviation > fail deviation
    # ---------------------------------

    binom_p = binomtest(
        wins,
        total,
        p=0.5,
        alternative="greater"
    ).pvalue

    # ---------------------------------
    # Wilcoxon Signed-Rank Test
    # H1:
    # median(diff) > 0
    # ---------------------------------

    try:

        wilcox_stat, wilcox_p = wilcoxon(
            diff,
            alternative="greater",
            zero_method="wilcox"
        )

        # ---------------------------------
        # Effect Size r
        # ---------------------------------

        z_score = norm.ppf(1 - wilcox_p)

        effect_r = z_score / np.sqrt(len(diff))

    except Exception:

        wilcox_stat = np.nan
        wilcox_p = np.nan
        effect_r = np.nan

    # ---------------------------------
    # Median Difference
    # ---------------------------------

    median_diff = np.median(diff)

    mean_diff = np.mean(diff)

    # ---------------------------------
    # Save
    # ---------------------------------

    records.append({
        "Metric":
            succ_col.replace(" abs(succ-avg)", ""),

        "Wins":
            wins,

        "Losses":
            losses,

        "Ties":
            ties,

        "Win Ratio":
            f"{wins}/{total} ({win_rate:.2%})",

        "Mean Diff":
            mean_diff,

        "Median Diff":
            median_diff,

        "Binomial p":
            binom_p,

        "Wilcoxon Statistic":
            wilcox_stat,

        "Wilcoxon p":
            wilcox_p,

        "Effect Size r":
            effect_r
    })

# =====================================================
# Result Table
# =====================================================

result = pd.DataFrame(records)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n")
print("=" * 100)
print("Statistical Validation Results")
print("=" * 100)

print(result)

# =====================================================
# Optional Sorting
# =====================================================

result = result.sort_values(
    by="Effect Size r",
    ascending=False
)

print("\n")
print("=" * 100)
print("Sorted by Effect Size")
print("=" * 100)

print(result)

# =====================================================
# Save
# =====================================================

result.to_excel(
    "RQ3_Statistical_Test.xlsx",
    index=False
)

print("\nSaved:")
print("RQ3_Statistical_Test.xlsx")