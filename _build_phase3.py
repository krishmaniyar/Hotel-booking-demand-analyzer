"""
_build_phase3.py
Appends Phase 3 cells (Tasks 10-15: Univariate Analysis) to ML_Ex01_EDA.ipynb.
Run: python _build_phase3.py
"""
import json
import textwrap

# ── Cell helpers ──────────────────────────────────────────────────────────────
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

# ── Load existing notebook ─────────────────────────────────────────────────────
NB_PATH = "ML_Ex01_EDA.ipynb"
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

last = nb["cells"][-1]
if last["cell_type"] == "markdown" and "Phase 3" in "".join(last["source"]):
    nb["cells"].pop()
    print("Removed Phase 3 continuation marker (will re-add Phase 4 at end).")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 10  — Numerical Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 10. Numerical Feature Analysis (Task 10)"
]))

new_cells.append(code("""
# ── 10.1 Define important numerical columns ──────────────────────────────────
IMPORTANT_NUM_COLS = [
    "lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights", 
    "adults", "children", "babies", "previous_cancellations", 
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", 
    "total_of_special_requests", "required_car_parking_spaces"
]

print(f"Defined IMPORTANT_NUM_COLS with {len(IMPORTANT_NUM_COLS)} features.")
"""))

new_cells.append(code("""
# ── 10.2 Compute numerical statistics ─────────────────────────────────────────
# Exclude is_canceled and identifier-like columns (agent, company).
# children is Int64, safe to cast to float for stats.

num_cols = train_df_deduped.select_dtypes(include="number").columns.tolist()
excluded = ["is_canceled", "agent", "company"]
num_cols = [c for c in num_cols if c not in excluded]

rows = []
for col in num_cols:
    # Dropna to avoid stat errors
    series = train_df_deduped[col].dropna().astype(float)
    
    count  = len(series)
    mean   = series.mean()
    median = series.median()
    mode   = series.mode().iloc[0] if not series.mode().empty else np.nan
    std    = series.std()
    var    = series.var()
    c_min  = series.min()
    c_max  = series.max()
    c_range= c_max - c_min
    q1     = series.quantile(0.25)
    q3     = series.quantile(0.75)
    iqr    = q3 - q1
    skew   = stats.skew(series)
    kurt   = stats.kurtosis(series)
    cv     = std / mean if mean != 0 else np.nan
    
    rows.append({
        "Feature": col,
        "Count": count,
        "Mean": round(mean, 2),
        "Median": round(median, 2),
        "Mode": round(mode, 2),
        "Std": round(std, 2),
        "Variance": round(var, 2),
        "Min": round(c_min, 2),
        "Max": round(c_max, 2),
        "Range": round(c_range, 2),
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
        "CV": round(cv, 2)
    })

num_stats_df = pd.DataFrame(rows)

print("=" * 65)
print("NUMERICAL FEATURE STATISTICS")
print("=" * 65)
display(num_stats_df.style.hide(axis="index"))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 11  — Numerical Feature Visualization
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 11. Numerical Feature Visualization (Task 11)"
]))

new_cells.append(code("""
# ── 11.1 Visualize IMPORTANT_NUM_COLS ─────────────────────────────────────────

def classify_skew(skew_val):
    if abs(skew_val) < 0.5:
        return "Symmetric"
    elif 0.5 <= abs(skew_val) < 1.0:
        return "Positively skewed" if skew_val > 0 else "Negatively skewed"
    else:
        return "Highly positively skewed" if skew_val > 0 else "Highly negatively skewed"

for col in IMPORTANT_NUM_COLS:
    series = train_df_deduped[col].dropna().astype(float)
    skew_val = num_stats_df.loc[num_stats_df["Feature"] == col, "Skewness"].values[0]
    skew_class = classify_skew(skew_val)
    
    # Check for extreme outliers manually for markdown description
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    has_outliers = (series < (q1 - 1.5 * iqr)).any() or (series > (q3 + 1.5 * iqr)).any()
    outlier_text = " (with visible outliers)" if has_outliers else ""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram + KDE
    sns.histplot(series, kde=True, ax=axes[0], color="skyblue", bins=30)
    axes[0].set_title(f"Histogram: {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")
    
    # Density plot
    sns.kdeplot(series, ax=axes[1], color="navy", fill=True)
    axes[1].set_title(f"Density Plot: {col}")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Density")
    
    # Box plot
    sns.boxplot(x=series, ax=axes[2], color="lightgreen")
    axes[2].set_title(f"Box Plot: {col}")
    axes[2].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()
    
    print(f"**Observation for {col}:** The distribution is {skew_class}{outlier_text} based on skewness = {skew_val}.")
    print("-" * 120)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 12  — Skewness Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 12. Skewness Analysis (Task 12)"
]))

new_cells.append(code("""
# ── 12.1 Classify and recommend transformations ───────────────────────────────

skew_rows = []
for idx, row in num_stats_df.iterrows():
    feature = row["Feature"]
    skew_val = row["Skewness"]
    c_min = row["Min"]
    abs_skew = abs(skew_val)
    
    if abs_skew < 0.5:
        classification = "Approximately symmetric"
        transform = "No transformation needed"
    elif abs_skew < 1.0:
        classification = "Moderately skewed"
        transform = "Square-root or Robust scaling"
    else:
        classification = "Highly skewed"
        if c_min >= 0:
            transform = "Log1p transformation"
        else:
            transform = "Yeo-Johnson transformation"
            
    skew_rows.append({
        "Feature": feature,
        "Skewness": skew_val,
        "Abs_Skewness": abs_skew,
        "Classification": classification,
        "Recommended Transformation": transform
    })

skew_df = pd.DataFrame(skew_rows).sort_values("Abs_Skewness", ascending=False).reset_index(drop=True)

print("=" * 65)
print("SKEWNESS ANALYSIS & TRANSFORMATION RECOMMENDATIONS")
print("=" * 65)
display(skew_df.drop(columns=["Abs_Skewness"]).style.hide(axis="index"))
"""))

new_cells.append(code("""
# ── 12.2 Visualization: Horizontal bar chart of absolute skewness ─────────────

fig, ax = plt.subplots(figsize=(10, max(4, len(skew_df) * 0.4)))

colors = []
for cls in skew_df["Classification"]:
    if "Highly" in cls:
        colors.append("indianred")
    elif "Moderately" in cls:
        colors.append("sandybrown")
    else:
        colors.append("mediumseagreen")

bars = ax.barh(
    skew_df["Feature"],
    skew_df["Abs_Skewness"],
    color=colors,
    edgecolor="white"
)

# Reference lines
ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="0.5 (Moderate threshold)")
ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.7, label="1.0 (High threshold)")

ax.set_title("|Skewness| of Numerical Features (Ranked)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Absolute Skewness", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.invert_yaxis()
ax.legend(loc="lower right")

# Custom legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='indianred', label='Highly skewed (|skew| >= 1.0)'),
    Patch(facecolor='sandybrown', label='Moderately skewed (0.5 <= |skew| < 1.0)'),
    Patch(facecolor='mediumseagreen', label='Approximately symmetric (|skew| < 0.5)')
]
ax.legend(handles=legend_elements, loc='lower right', title="Classification")

ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 13  — Categorical Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 13. Categorical Feature Analysis (Task 13)"
]))

new_cells.append(code("""
# ── 13.1 Define categorical columns and compute stats ─────────────────────────
CAT_COLS = [
    "hotel", "meal", "country", "market_segment", "distribution_channel", 
    "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type"
]
n_total = len(train_df_deduped)

cat_rows = []
for col in CAT_COLS:
    vc = train_df_deduped[col].value_counts(dropna=True)
    nunique = len(vc)
    
    if nunique > 0:
        most_freq_cat = vc.index[0]
        most_freq_count = vc.iloc[0]
        most_freq_pct = (most_freq_count / n_total) * 100
        
        least_freq_cat = vc.index[-1]
        least_freq_count = vc.iloc[-1]
    else:
        most_freq_cat, most_freq_count, most_freq_pct = "N/A", 0, 0
        least_freq_cat, least_freq_count = "N/A", 0
        
    cardinality_ratio = nunique / n_total
    
    cat_rows.append({
        "Feature": col,
        "Unique Categories": nunique,
        "Most Frequent": most_freq_cat,
        "Most Freq Count": most_freq_count,
        "Most Freq %": round(most_freq_pct, 2),
        "Least Frequent": least_freq_cat,
        "Least Freq Count": least_freq_count,
        "Cardinality Ratio": round(cardinality_ratio, 6)
    })

cat_stats_df = pd.DataFrame(cat_rows)

print("=" * 65)
print("CATEGORICAL FEATURE SUMMARY")
print("=" * 65)
display(cat_stats_df.style.hide(axis="index"))
"""))

new_cells.append(code("""
# ── 13.2 Frequency tables and bar plots ───────────────────────────────────────
for col in CAT_COLS:
    print(f"\\n{'='*50}\\nFeature: {col}\\n{'='*50}")
    
    vc = train_df_deduped[col].value_counts(dropna=False)
    vc_pct = train_df_deduped[col].value_counts(dropna=False, normalize=True) * 100
    
    freq_df = pd.DataFrame({"Count": vc, "%": vc_pct.round(2)})
    
    if col == "country":
        display(freq_df.head(15))
        remaining = len(freq_df) - 15
        if remaining > 0:
            print(f"... + {remaining} other countries")
    else:
        display(freq_df)
        
        # Bar plot (skip country as requested)
        plt.figure(figsize=(8, 4))
        sns.countplot(data=train_df_deduped, y=col, order=vc.index, palette="viridis")
        plt.title(f"Category Frequencies: {col}", fontweight="bold")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 14  — Rare Category Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 14. Rare Category Analysis (Task 14)"
]))

new_cells.append(code("""
# ── 14.1 Identify rare categories ─────────────────────────────────────────────
n_total = len(train_df_deduped)
rare_rows = []

for col in CAT_COLS:
    vc_pct = train_df_deduped[col].value_counts(normalize=True) * 100
    
    under_1_pct = vc_pct[vc_pct < 1.0].index.tolist()
    under_5_pct = vc_pct[vc_pct < 5.0].index.tolist()
    
    examples = under_1_pct[:3] if len(under_1_pct) > 0 else (under_5_pct[:3] if len(under_5_pct) > 0 else ["None"])
    
    rare_rows.append({
        "Feature": col,
        "Categories < 1%": len(under_1_pct),
        "Categories < 5%": len(under_5_pct),
        "Example Rare Categories": ", ".join(map(str, examples))
    })

rare_df = pd.DataFrame(rare_rows)

print("=" * 65)
print("RARE CATEGORY ANALYSIS (<1% and <5%)")
print("=" * 65)
display(rare_df.style.hide(axis="index"))
"""))

new_cells.append(md("""## Rare Category Recommendations

*   **hotel**: No rare categories. **Retain as is.**
*   **meal**: 'Undefined' and 'FB' are rare. **Combine into "Other"** to reduce noise, since standard plans (BB, HB) dominate.
*   **country**: 146 categories <1%. **Target encode or group into "Top N" + "Other"**. Frequency encoding could also work, but grouping is simpler.
*   **market_segment**: 'Undefined', 'Aviation', 'Complementary' are <1%. **Combine into "Other"** as they don't represent the core business volume.
*   **distribution_channel**: 'Undefined' and 'GDS' are rare. **Combine into "Other"**.
*   **reserved_room_type / assigned_room_type**: Many letter codes are <1%. **Group all codes outside top 5 into "Other"** to preserve major types and reduce cardinality before one-hot encoding.
*   **deposit_type**: 'Refundable' is <1%. **Retain or group with No Deposit**, depending on business logic (Refundable operates similarly to No Deposit for risk).
*   **customer_type**: 'Group' is <1%. **Combine into "Other" or retain** (might have specific cancellation logic). Retain for now.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 15  — High-Cardinality Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 15. High-Cardinality Feature Analysis (Task 15)"
]))

new_cells.append(code("""
# ── 15.1 High-cardinality stats (country, agent, company) ─────────────────────
hc_cols = ["country", "agent", "company"]
hc_df_copy = train_df_deduped[hc_cols].copy()

# Cast agent and company to string, fillna with Unknown
hc_df_copy["agent"] = hc_df_copy["agent"].astype(str).replace("<NA>", "Unknown")
hc_df_copy["company"] = hc_df_copy["company"].astype(str).replace("<NA>", "Unknown")
hc_df_copy["country"] = hc_df_copy["country"].fillna("Unknown")

n_total = len(hc_df_copy)

hc_rows = []
for col in hc_cols:
    nunique = hc_df_copy[col].nunique()
    card_ratio = nunique / n_total
    hc_rows.append({
        "Feature": col,
        "Unique Categories (inc. Unknown)": nunique,
        "Cardinality Ratio": round(card_ratio, 6)
    })

display(pd.DataFrame(hc_rows).style.hide(axis="index"))
"""))

new_cells.append(code("""
# ── 15.2 Bar charts for Top 15 categories ─────────────────────────────────────
for col in hc_cols:
    vc = hc_df_copy[col].value_counts().head(15)
    
    plt.figure(figsize=(10, 4))
    sns.barplot(x=vc.values, y=vc.index, palette="magma")
    plt.title(f"Top 15 Categories for High-Cardinality Feature: {col}", fontweight="bold")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
"""))

new_cells.append(md("""## Why One-Hot Encoding is Unsuitable for High Cardinality

Applying one-hot encoding to features like `country`, `agent`, and `company` is fundamentally a bad idea due to their high unique value counts:
- **`country`** has ~160 unique categories.
- **`agent`** has ~320 unique categories.
- **`company`** has ~340 unique categories.

If one-hot encoded, these three columns alone would add over **800 new sparse binary columns** to the dataset. This leads to the "curse of dimensionality", drastically increasing memory usage, slowing down model training, and significantly raising the risk of overfitting (especially for tree-based models where splits become heavily fragmented).

### Recommended Alternatives:
1. **Target Encoding**: Replace categories with the mean of the target (`is_canceled`). This collapses the categorical information into a single dense numerical column. **Caveat**: Must be computed only on the training folds and applied to validation/test to prevent data leakage.
2. **Frequency/Count Encoding**: Replace categories with their occurrence count. Useful if rare vs frequent implies different cancellation behaviors.
3. **Grouping (Top-N + "Other")**: Keep only the top 10-15 countries/agents and group the rest into an "Other" bucket. Then apply one-hot encoding on the reduced set (e.g., 15 columns instead of 160).
"""))

# ── Phase 4 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 4 — Target Variable and Outlier Analysis (Tasks 16-18)"
]))

# ── Append to notebook and write ──────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print(f"Phase 3 appended successfully.")
print(f"  New cells added : {added}")
print(f"  Total cells now : {total}")
print(f"  Written to      : {NB_PATH}")
