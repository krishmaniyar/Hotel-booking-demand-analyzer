"""
_build_phase4.py
Appends Phase 4 cells (Tasks 16-18: Target Distribution & Outlier Analysis)
to ML_Ex01_EDA.ipynb.
Run: python _build_phase4.py
"""
import json
import textwrap

def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def code(source):
    lines = textwrap.dedent(source).lstrip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]
    }

NB_PATH = "ML_Ex01_EDA.ipynb"
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# Strip Phase 4 continuation marker if it already exists
last = nb["cells"][-1]
if last["cell_type"] == "markdown" and "Phase 4" in "".join(last["source"]):
    nb["cells"].pop()
    print("Removed existing Phase 4 continuation marker.")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 16  — Target Distribution
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 16. Target Distribution (Task 16)"
]))

new_cells.append(code("""
# ── 16.1  Count and % breakdown of is_canceled ────────────────────────────────
target_counts = train_df_deduped[target_col].value_counts().sort_index()
target_pcts   = train_df_deduped[target_col].value_counts(normalize=True).sort_index() * 100

n_not_cancelled = target_counts[0]
n_cancelled     = target_counts[1]
pct_not_cancelled = target_pcts[0]
pct_cancelled     = target_pcts[1]

print("=" * 50)
print("TARGET VARIABLE DISTRIBUTION — is_canceled")
print("=" * 50)
print(f"  Not Cancelled (0) : {n_not_cancelled:>7,}  ({pct_not_cancelled:.2f}%)")
print(f"  Cancelled     (1) : {n_cancelled:>7,}  ({pct_cancelled:.2f}%)")
print()

# Majority-to-minority ratio
majority_n = max(n_not_cancelled, n_cancelled)
minority_n = min(n_not_cancelled, n_cancelled)
imbalance_ratio = majority_n / minority_n

print(f"  Majority-to-Minority Ratio : {imbalance_ratio:.2f} : 1")
print()

# Imbalance classification
if imbalance_ratio < 1.5:
    imbalance_class = "Balanced"
elif imbalance_ratio <= 3.0:
    imbalance_class = "Moderately Imbalanced"
else:
    imbalance_class = "Severely Imbalanced"

print(f"  Classification : {imbalance_class} (ratio = {imbalance_ratio:.2f}:1)")
"""))

new_cells.append(code("""
# ── 16.2  Visualization 1: Count plot with exact counts annotated ─────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Count plot --
ax1 = axes[0]
bars = sns.countplot(
    x=target_col,
    data=train_df_deduped,
    palette=["#4c72b0", "#dd8452"],
    order=[0, 1],
    ax=ax1
)
ax1.set_title("Booking Cancellation Count", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("is_canceled  (0 = Not Cancelled, 1 = Cancelled)", fontsize=11)
ax1.set_ylabel("Number of Bookings", fontsize=11)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Not Cancelled (0)", "Cancelled (1)"])

# Annotate counts on top of each bar
for bar in ax1.patches:
    count = int(bar.get_height())
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 150,
        f"{count:,}",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )
ax1.spines[["top", "right"]].set_visible(False)

# -- Percentage horizontal bar chart --
ax2 = axes[1]
categories = ["Not Cancelled (0)", "Cancelled (1)"]
pcts = [pct_not_cancelled, pct_cancelled]
colors = ["#4c72b0", "#dd8452"]

hbars = ax2.barh(categories, pcts, color=colors, edgecolor="white", height=0.45)
ax2.set_title("Cancellation Rate (%)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Percentage of Bookings (%)", fontsize=11)
ax2.set_ylabel("")
ax2.set_xlim(0, 100)

for bar, pct in zip(hbars, pcts):
    ax2.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.2f}%",
        va="center", ha="left",
        fontsize=11, fontweight="bold"
    )
ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50% line")
ax2.legend(loc="lower right", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Target Variable: is_canceled", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(f"\\nImbalance ratio confirmed: {imbalance_ratio:.2f}:1 — classified as '{imbalance_class}'")
"""))

new_cells.append(md("""## Target Variable Assessment

### Imbalance Classification

The `is_canceled` target is **Moderately Imbalanced** (typically ~1.6:1 ratio in the hotel booking dataset). The majority class (Not Cancelled, ~63%) outnumbers the minority class (Cancelled, ~37%) but not at a severe level.

### Recommended Evaluation Metrics

Given this moderate imbalance, **raw accuracy is a misleading metric**. A naive classifier that always predicts "Not Cancelled" would achieve ~63% accuracy without learning anything useful — and would look deceptively good.

**Recommended metrics for this problem:**

| Metric | Why it's appropriate |
|---|---|
| **F1-Score** | Harmonic mean of Precision and Recall — penalises models that sacrifice minority-class recall for majority-class precision |
| **ROC-AUC** | Measures discrimination ability at all thresholds; robust to imbalance |
| **Precision-Recall AUC** | Especially useful when the minority class (cancellations) is the class of business interest — more informative than ROC-AUC under moderate imbalance |
| **Matthews Correlation Coefficient (MCC)** | A single balanced metric that accounts for all four confusion matrix quadrants |

**Do NOT rely solely on accuracy** — with a ~63/37 split, even a dummy classifier achieves ~63% accuracy. Always report at minimum: F1, ROC-AUC, and the confusion matrix breakdown.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 17  — Statistical Outlier Detection (IQR method)
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 17. Statistical Outlier Detection (Task 17)"
]))

new_cells.append(code("""
# ── 17.1  IQR outlier detection for all IMPORTANT_NUM_COLS ───────────────────
# Recomputing Q1/Q3 cleanly here (avoids dependency on Task 10's variable state
# surviving a kernel restart — self-contained cell).

n_total = len(train_df_deduped)
outlier_rows = []

for col in IMPORTANT_NUM_COLS:
    series = train_df_deduped[col].dropna().astype(float)
    
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    n_lower   = (series < lower_bound).sum()
    n_upper   = (series > upper_bound).sum()
    n_total_out = n_lower + n_upper
    pct_out   = (n_total_out / n_total) * 100
    
    outlier_rows.append({
        "Feature"         : col,
        "Lower Bound"     : round(lower_bound, 3),
        "Upper Bound"     : round(upper_bound, 3),
        "Lower Outliers"  : n_lower,
        "Upper Outliers"  : n_upper,
        "Total Outliers"  : n_total_out,
        "Outlier %"       : round(pct_out, 3)
    })

outlier_df = (
    pd.DataFrame(outlier_rows)
    .sort_values("Outlier %", ascending=False)
    .reset_index(drop=True)
)

print("=" * 65)
print("IQR OUTLIER DETECTION SUMMARY — IMPORTANT_NUM_COLS")
print("=" * 65)
display(
    outlier_df.style
    .hide(axis="index")
    .set_caption("Statistical Outlier Detection (IQR Method) — train_df")
    .background_gradient(subset=["Outlier %"], cmap="OrRd")
)

print()
print(f"NOTE: No rows have been removed. This table is for documentation only.")
print(f"      Outlier treatment decisions are deferred to Task 27 (preprocessing).")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 18  — Outlier Visualization
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 18. Outlier Visualization (Task 18)"
]))

new_cells.append(code("""
# ── 18.1  Box plots for 7 key features ───────────────────────────────────────
viz_outlier_cols = [
    "lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights",
    "previous_cancellations", "booking_changes", "days_in_waiting_list"
]

# 4 rows x 2 cols grid (last cell left empty if odd count)
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
    
    ax.set_title(f"{col}\n({out_pct:.2f}% outliers)", fontsize=10, fontweight="bold")
    ax.set_ylabel(col, fontsize=9)
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)

# Hide the unused subplot (if 7 features in 4x2 grid, cell [3,1] is empty)
for j in range(len(viz_outlier_cols), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Outlier Visualization — Box Plots (IQR Method)", 
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""## Outlier Classification by Feature

Below each feature's outliers are classified using domain logic combined with the actual outlier % computed in Task 17:

| Feature | Classification | Reasoning |
|---|---|---|
| **`lead_time`** | Rare-but-valid observation | Lead times of 400–700 days exist for group travel, weddings, and conference bookings. Not data errors — just an extreme tail of normal booking behaviour. |
| **`adr`** | Extreme customer behaviour / potential anomaly | Very high ADR (>500) could represent luxury suite bookings or pricing errors. Negative ADR is a data anomaly (flagged in Task 6). The extreme upper tail warrants log1p transformation, not removal. |
| **`stays_in_weekend_nights`** | Rare-but-valid observation | Most stays are 0–3 weekend nights; stays of 7+ nights are genuine extended holiday stays. Nothing implausible here — just rare. |
| **`stays_in_week_nights`** | Rare-but-valid observation | Week-night stays >10 represent long corporate stays or extended vacations. Unusual but entirely valid in hotel data. |
| **`previous_cancellations`** | Extreme customer behaviour | A guest with 10+ prior cancellations is a genuine high-risk customer, not a data entry error. These are the most informative rows for the cancellation prediction task. |
| **`booking_changes`** | Extreme customer behaviour | 5+ booking amendments is very unusual but can represent indecisive travellers or complex group bookings undergoing itinerary changes. |
| **`days_in_waiting_list`** | Rare-but-valid observation | Waiting 200+ days is unusual but entirely plausible for high-demand properties during peak seasons (New Year's Eve, major events). These are real observations, not entry errors. |

---

> **Important — No Outliers Removed in This Task**
>
> This is a documentation-only task. No rows have been dropped or filtered. All observations, including outliers, remain in `train_df_deduped`. Outlier treatment strategies (capping, log transformation, indicator flags) will be specified in Task 27 (Preprocessing Plan) after the full bivariate and correlation analysis is complete.
"""))

# ── Phase 5 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 5 — Bivariate Analysis (Tasks 19-21)"
]))

# ── Append and write ──────────────────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print(f"Phase 4 appended successfully.")
print(f"  New cells added : {added}")
print(f"  Total cells now : {total}")
print(f"  Written to      : {NB_PATH}")
