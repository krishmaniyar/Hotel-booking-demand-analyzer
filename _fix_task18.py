"""
_fix_task18.py
Patches the Task 18 box-plot cell in ML_Ex01_EDA.ipynb to fix the
f-string newline SyntaxError (literal \n inside f-string).
"""
import json, textwrap

NB_PATH = "ML_Ex01_EDA.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Find the Task 18 box-plot code cell by a unique snippet
TARGET_SNIPPET = "viz_outlier_cols"
fix_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and TARGET_SNIPPET in "".join(cell["source"]):
        fix_idx = i
        print(f"Found Task 18 box-plot cell at index {i}")
        break

assert fix_idx is not None, "Could not find Task 18 box-plot cell!"

FIXED_SOURCE = textwrap.dedent("""\
# ── 18.1  Box plots for 7 key features ───────────────────────────────────────
viz_outlier_cols = [
    "lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights",
    "previous_cancellations", "booking_changes", "days_in_waiting_list"
]

# 4 rows x 2 cols grid (last cell left empty for odd count)
n_cols = 2
n_rows = (len(viz_outlier_cols) + n_cols - 1) // n_cols   # = 4

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes_flat = axes.flatten()

for i, col in enumerate(viz_outlier_cols):
    ax = axes_flat[i]
    series = train_df_deduped[col].dropna().astype(float)

    sns.boxplot(
        y=series,
        ax=ax,
        color="#5b84b1",
        flierprops=dict(
            marker="o", markerfacecolor="#dd8452",
            markersize=3, alpha=0.4, linestyle="none"
        )
    )

    # Retrieve outlier % from outlier_df computed in Task 17
    row = outlier_df[outlier_df["Feature"] == col]
    out_pct = row["Outlier %"].values[0] if len(row) > 0 else 0.0

    title_text = col + " (" + str(round(out_pct, 2)) + "% outliers)"
    ax.set_title(title_text, fontsize=10, fontweight="bold")
    ax.set_ylabel(col, fontsize=9)
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)

# Hide the unused subplot (7 features in 4x2 grid -> cell [3,1] empty)
for j in range(len(viz_outlier_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Outlier Visualization - Box Plots (IQR Method)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
""")

lines = FIXED_SOURCE.split("\n")
nb["cells"][fix_idx]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
nb["cells"][fix_idx]["outputs"] = []
nb["cells"][fix_idx]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Task 18 cell patched successfully — f-string newline fixed.")
