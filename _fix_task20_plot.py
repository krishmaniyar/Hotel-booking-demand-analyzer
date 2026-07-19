"""
_fix_task20_plot.py
Patches the Task 20 box+violin cell to remove the literal \\n in set_title.
Uses ax.set_title with only one line, moves subtitle to ax.text underneath.
"""
import json, textwrap

NB_PATH = "ML_Ex01_EDA.ipynb"
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

TARGET = "VIZ_NUM_TARGET"
fix_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and TARGET in "".join(cell["source"]):
        fix_idx = i
        print("Found Task 20 plot cell at index", i)
        break

assert fix_idx is not None, "Could not find Task 20 plot cell!"

FIXED = textwrap.dedent("""\
# ── 20.2  Paired box + violin plots for 6 key features ───────────────────────
VIZ_NUM_TARGET = [
    "lead_time", "adr", "previous_cancellations",
    "total_of_special_requests", "required_car_parking_spaces", "booking_changes"
]

n_features = len(VIZ_NUM_TARGET)
fig, axes = plt.subplots(n_features, 2, figsize=(14, n_features * 3.2))

for i, col in enumerate(VIZ_NUM_TARGET):
    pct_row = target_stats_df[target_stats_df["Feature"] == col]
    pct_val = pct_row["% Diff in Means"].values[0] if len(pct_row) > 0 else 0.0
    mean0   = pct_row["Mean (Not Cancelled)"].values[0] if len(pct_row) > 0 else 0.0
    mean1   = pct_row["Mean (Cancelled)"].values[0] if len(pct_row) > 0 else 0.0

    plot_df = train_df_deduped[[col, target_col]].dropna().copy()
    plot_df[col] = plot_df[col].astype(float)
    plot_df["Status"] = plot_df[target_col].map({0: "Not Cancelled", 1: "Cancelled"})

    title_str    = col + "  (mean diff: " + str(round(pct_val, 1)) + "%)"
    subtitle_str = "Not Cancelled mean=" + str(round(mean0, 2)) + "  |  Cancelled mean=" + str(round(mean1, 2))

    # Box plot
    ax_box = axes[i, 0]
    sns.boxplot(
        data=plot_df, x="Status", y=col,
        palette={"Not Cancelled": "#4c72b0", "Cancelled": "#dd8452"},
        ax=ax_box,
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=2, alpha=0.3)
    )
    ax_box.set_title(title_str, fontsize=9, fontweight="bold")
    ax_box.set_xlabel(subtitle_str, fontsize=7.5)
    ax_box.set_ylabel(col, fontsize=9)
    ax_box.spines[["top", "right"]].set_visible(False)

    # Violin plot
    ax_vln = axes[i, 1]
    sns.violinplot(
        data=plot_df, x="Status", y=col,
        palette={"Not Cancelled": "#4c72b0", "Cancelled": "#dd8452"},
        inner="quartile",
        ax=ax_vln
    )
    ax_vln.set_title(title_str + " - density view", fontsize=9, fontweight="bold")
    ax_vln.set_xlabel("")
    ax_vln.set_ylabel(col, fontsize=9)
    ax_vln.spines[["top", "right"]].set_visible(False)

plt.suptitle("Numerical Features vs is_canceled (Box + Violin)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
plt.show()
""")

lines = FIXED.split("\n")
nb["cells"][fix_idx]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
nb["cells"][fix_idx]["outputs"] = []
nb["cells"][fix_idx]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Task 20 plot cell patched — literal newline in set_title removed.")
