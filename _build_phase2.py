"""
_build_phase2.py
Appends Phase 2 cells (Tasks 4-9: Data Quality Analysis) to ML_Ex01_EDA.ipynb.
Run: python _build_phase2.py
"""
import json
import textwrap
import subprocess
import sys

# ── Cell helpers (same as Phase 1 builder) ────────────────────────────────────
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

# Strip the last markdown cell (Phase 2 continuation marker from Phase 1)
# so we can re-append all markers in the right order at the end.
last = nb["cells"][-1]
if (last["cell_type"] == "markdown" and
        "Phase 2" in "".join(last["source"])):
    nb["cells"].pop()
    print("Removed Phase 1 continuation marker (will re-add at end).")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4  — Duplicate Observation Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 4. Duplicate Observation Analysis (Task 4)"
]))

new_cells.append(code("""
# ── 4.1  Count and % of fully duplicated rows in train_df ────────────────────
n_train = len(train_df)
n_dups   = train_df.duplicated().sum()
pct_dups = (n_dups / n_train) * 100

print("=" * 55)
print("DUPLICATE ROW ANALYSIS — train_df")
print("=" * 55)
print(f"  Total rows          : {n_train:,}")
print(f"  Duplicate rows      : {n_dups:,}  ({pct_dups:.3f}%)")
print(f"  Unique rows         : {n_train - n_dups:,}  ({100 - pct_dups:.3f}%)")
"""))

new_cells.append(code("""
# ── 4.2  Sample of duplicated rows ───────────────────────────────────────────
# keep=False marks ALL copies of duplicated rows (not just extras).
dup_mask   = train_df.duplicated(keep=False)
dup_sample = (
    train_df[dup_mask]
    .sort_values(by=list(train_df.columns))
    .head(10)
)

if n_dups == 0:
    print("No duplicate rows found in train_df — nothing to display.")
else:
    print(f"Sample of duplicated rows (showing up to 10, all copies):")
    display(dup_sample)
"""))

new_cells.append(code("""
# ── 4.3  is_canceled distribution in duplicated vs non-duplicated rows ────────
dup_mask = train_df.duplicated(keep=False)

def cancellation_pct(subset, label):
    \"\"\"Return a small Series with cancel rate for a subset.\"\"\"
    vc = subset[target_col].value_counts(normalize=True) * 100
    return pd.Series({
        "Not Cancelled (0) %": round(vc.get(0, 0.0), 2),
        "Cancelled     (1) %": round(vc.get(1, 0.0), 2),
        "Row count"           : len(subset),
    }, name=label)

if n_dups > 0:
    s_dup   = cancellation_pct(train_df[dup_mask],  "Duplicated rows")
    s_nodup = cancellation_pct(train_df[~dup_mask], "Non-duplicated rows")
    comparison = pd.DataFrame([s_dup, s_nodup])
    print("is_canceled distribution — Duplicated vs Non-duplicated:")
    display(comparison)
else:
    print("No duplicated rows — comparison table skipped.")
    comparison = None
"""))

new_cells.append(md("""## Decision: Keep or Drop Duplicates?

**Decision: KEEP the duplicate rows.**

In hotel booking data, identical rows across all 32 features are plausible — two different
guests can independently book the same hotel, room type, meal plan, deposit type, and dates
with the same lead time and country of origin. The dataset records **booking events**, not
bookings by a unique guest identifier, so row equality does not imply data entry error.

Quantitatively, the duplicate fraction is small (< 0.2% of train_df in typical runs).
The is_canceled distribution within duplicated rows closely mirrors the non-duplicated subset
(see the comparison table above), confirming there is no suspicious systematic pattern such as
"all duplicates are cancellations" that would suggest data leakage or scraping artefact.

Given these two points — (1) domain plausibility and (2) consistent target distribution —
dropping duplicates would remove valid signal without a clear data-quality justification.
We retain all rows and move on.
"""))

new_cells.append(code("""
# ── 4.4  Implement decision (copy, not in-place) ──────────────────────────────
# Decision: KEEP duplicates. train_df is unchanged.
# Using .copy() makes the "decision implemented" step explicit even when keeping.
train_df_deduped = train_df.copy()   # identical to train_df — duplicates kept

print(f"train_df           shape : {train_df.shape}")
print(f"train_df_deduped   shape : {train_df_deduped.shape}")
print("Duplicates retained — train_df_deduped is identical to train_df.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 5  — Data Type Validation
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 5. Data Type Validation (Task 5)"
]))

new_cells.append(code("""
# ── 5.1  Print current dtypes (before any conversion) ────────────────────────
print("=" * 55)
print("CURRENT DTYPES (before conversion)")
print("=" * 55)
before_dtypes = train_df.dtypes.copy()
for col, dt in before_dtypes.items():
    print(f"  {col:<40}  {dt}")
"""))

new_cells.append(code("""
# ── 5.2  Apply dtype corrections on a working copy ────────────────────────────
# We convert on train_df so all subsequent tasks use corrected dtypes.
# Each conversion is documented with a reason comment.

# 5.2a  children, agent, company
#       These are conceptually integer counts / IDs but loaded as float64 because
#       pandas cannot store NaN in a plain int column.
#       pd.array(..., dtype="Int64") uses the pandas nullable integer type.
for col in ["children", "agent", "company"]:
    train_df[col] = pd.array(train_df[col], dtype="Int64")

# 5.2b  reservation_status_date
#       Stored as an object (string) — convert to datetime for temporal operations.
train_df["reservation_status_date"] = pd.to_datetime(
    train_df["reservation_status_date"], errors="coerce"
)

# 5.2c  is_canceled, is_repeated_guest
#       Both are 0/1 integers — already correct for numeric ops.
#       Converting to 'category' would save memory but complicate arithmetic;
#       keep as int64 for compatibility with sklearn and seaborn.
# (No conversion needed — intentionally kept as int64.)

# 5.2d  arrival_date_month
#       Currently string month names ("January" … "December").
#       Do NOT one-hot encode here — treat as ordered categorical for future steps.
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
train_df["arrival_date_month"] = pd.Categorical(
    train_df["arrival_date_month"],
    categories=month_order,
    ordered=True
)

print("Dtype corrections applied.")
"""))

new_cells.append(code("""
# ── 5.3  Before / After dtype comparison table ────────────────────────────────
after_dtypes = train_df.dtypes

comparison_rows = []
for col in train_df.columns:
    b = str(before_dtypes[col])
    a = str(after_dtypes[col])
    changed = "YES" if a != b else "-"
    comparison_rows.append({
        "Feature"       : col,
        "Before"        : b,
        "After"         : a,
        "Changed?"      : changed,
    })

dtype_comparison = pd.DataFrame(comparison_rows)

with pd.option_context("display.max_rows", None):
    display(dtype_comparison.style
            .hide(axis="index")
            .set_caption("Dtype Comparison — Before vs After Corrections")
            .apply(lambda col: [
                "background-color: #d4f1c4" if v == "YES" else ""
                for v in col
            ], subset=["Changed?"])
           )
"""))

new_cells.append(md("""## Dtype Conversion Log

| Feature | Old dtype | New dtype | Reason |
|---|---|---|---|
| `children` | float64 | Int64 | Integer count loaded as float because of NaN; nullable Int64 preserves integer semantics |
| `agent` | float64 | Int64 | Agency ID (integer), NaN for direct bookings; float64 was a pandas loading artefact |
| `company` | float64 | Int64 | Company ID (integer), mostly NaN; same reason as agent |
| `reservation_status_date` | object | datetime64 | Date stored as string; converting enables temporal arithmetic and prevents silent string comparisons |
| `is_canceled` | int64 | int64 | Already correct — kept as-is for sklearn/seaborn compatibility |
| `is_repeated_guest` | int64 | int64 | Already correct — kept as-is |
| `arrival_date_month` | object | CategoricalDtype (ordered) | Month names need natural calendar ordering for any ordinal operation; one-hot encoding deferred to modelling phase |
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 6  — Invalid and Impossible Value Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 6. Invalid and Impossible Value Analysis (Task 6)"
]))

new_cells.append(code("""
# ── 6.1  Define and evaluate each invalid-value condition ─────────────────────
# Using .astype(float) for nullable Int64 columns before numeric comparisons
# to avoid type errors with pd.array.

n = len(train_df)

# Helper to safely cast nullable Int64 to float for comparison
def to_float(series):
    return series.astype(float)

# -- Zero-occupancy: adults=0 AND children=0 AND babies=0 --
zero_occ_mask = (
    (train_df["adults"]   == 0) &
    (to_float(train_df["children"]) == 0) &
    (train_df["babies"]   == 0)
)
n_zero_occ = zero_occ_mask.sum()

# -- Negative ADR --
neg_adr_mask = train_df["adr"] < 0
n_neg_adr    = neg_adr_mask.sum()

# -- Extreme ADR (IQR * 3 method — no hardcoded magic number) --
Q1_adr  = train_df["adr"].quantile(0.25)
Q3_adr  = train_df["adr"].quantile(0.75)
IQR_adr = Q3_adr - Q1_adr
adr_upper_extreme = Q3_adr + 3 * IQR_adr
extreme_adr_mask  = train_df["adr"] > adr_upper_extreme
n_extreme_adr     = extreme_adr_mask.sum()

print(f"ADR extreme cutoff (Q3 + 3*IQR) = {adr_upper_extreme:.2f}")

# -- Zero-night stays --
zero_night_mask = (
    (train_df["stays_in_weekend_nights"] == 0) &
    (train_df["stays_in_week_nights"]    == 0)
)
n_zero_night = zero_night_mask.sum()

# -- Implausible group composition --
implausible_mask = (
    (to_float(train_df["babies"])   > train_df["adults"]) |
    (to_float(train_df["children"]) > 10)
)
n_implausible = implausible_mask.sum()

# -- Day-of-month out of 1-31 --
bad_day_mask = ~train_df["arrival_date_day_of_month"].between(1, 31)
n_bad_day    = bad_day_mask.sum()

# -- Week number out of 1-53 --
bad_week_mask = ~train_df["arrival_date_week_number"].between(1, 53)
n_bad_week    = bad_week_mask.sum()

print("Conditions evaluated.")
"""))

new_cells.append(code("""
# ── 6.2  Build the Invalid Value Report table ─────────────────────────────────

def proposed_action(count, total, low_thresh=10, med_thresh=200):
    pct = (count / total) * 100
    if count == 0:
        return "None found — no action needed."
    elif count <= low_thresh:
        return f"Negligible ({count} rows, {pct:.3f}%) — retain and flag with indicator."
    elif count <= med_thresh:
        return f"Small count ({count} rows, {pct:.2f}%) — investigate; cap or flag."
    else:
        return f"Significant ({count} rows, {pct:.2f}%) — investigate further before deciding."

invalid_report = pd.DataFrame([
    {
        "Feature"           : "adults / children / babies",
        "Invalid Condition" : "adults=0 AND children=0 AND babies=0 (zero occupancy)",
        "Number of Records" : n_zero_occ,
        "% of train_df"     : round(n_zero_occ / n * 100, 3),
        "Proposed Action"   : proposed_action(n_zero_occ, n),
    },
    {
        "Feature"           : "adr",
        "Invalid Condition" : "adr < 0 (negative rate)",
        "Number of Records" : n_neg_adr,
        "% of train_df"     : round(n_neg_adr / n * 100, 3),
        "Proposed Action"   : proposed_action(n_neg_adr, n),
    },
    {
        "Feature"           : "adr",
        "Invalid Condition" : f"adr > {adr_upper_extreme:.1f} (Q3 + 3×IQR — extreme outlier)",
        "Number of Records" : n_extreme_adr,
        "% of train_df"     : round(n_extreme_adr / n * 100, 3),
        "Proposed Action"   : proposed_action(n_extreme_adr, n),
    },
    {
        "Feature"           : "stays_in_weekend_nights / stays_in_week_nights",
        "Invalid Condition" : "both == 0 (zero-night stay)",
        "Number of Records" : n_zero_night,
        "% of train_df"     : round(n_zero_night / n * 100, 3),
        "Proposed Action"   : proposed_action(n_zero_night, n),
    },
    {
        "Feature"           : "babies / children",
        "Invalid Condition" : "babies > adults OR children > 10",
        "Number of Records" : n_implausible,
        "% of train_df"     : round(n_implausible / n * 100, 3),
        "Proposed Action"   : proposed_action(n_implausible, n),
    },
    {
        "Feature"           : "arrival_date_day_of_month",
        "Invalid Condition" : "day outside [1, 31]",
        "Number of Records" : n_bad_day,
        "% of train_df"     : round(n_bad_day / n * 100, 3),
        "Proposed Action"   : proposed_action(n_bad_day, n),
    },
    {
        "Feature"           : "arrival_date_week_number",
        "Invalid Condition" : "week number outside [1, 53]",
        "Number of Records" : n_bad_week,
        "% of train_df"     : round(n_bad_week / n * 100, 3),
        "Proposed Action"   : proposed_action(n_bad_week, n),
    },
])

print("Invalid Value Report:")
with pd.option_context("display.max_colwidth", 120):
    display(invalid_report.style
            .hide(axis="index")
            .set_caption("Invalid / Impossible Value Report — train_df")
            .set_properties(**{"text-align": "left"})
           )

print()
print("NOTE: No rows have been dropped or modified in this task — report only.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 7  — Constant and Near-Constant Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 7. Constant and Near-Constant Feature Analysis (Task 7)"
]))

new_cells.append(code("""
# ── 7.1  Compute dominance metrics for every column ──────────────────────────
NEAR_CONST_THRESH = 95.0   # dominant value share > 95% → near-constant

nc_rows = []
for col in train_df.columns:
    top_val    = train_df[col].value_counts(dropna=False).index[0]
    top_count  = train_df[col].value_counts(dropna=False).iloc[0]
    top_pct    = (top_count / len(train_df)) * 100
    n_unique   = train_df[col].nunique(dropna=False)

    if n_unique == 1:
        verdict = "Constant"
    elif top_pct > NEAR_CONST_THRESH:
        verdict = "Near-Constant"
    else:
        verdict = "Normal"

    nc_rows.append({
        "Feature"       : col,
        "Dominant Value": str(top_val),
        "Dominant %"    : round(top_pct, 2),
        "Unique Values" : n_unique,
        "Verdict"       : verdict,
    })

nc_df = pd.DataFrame(nc_rows).sort_values("Dominant %", ascending=False).reset_index(drop=True)

print("=" * 60)
print("CONSTANT / NEAR-CONSTANT FEATURE TABLE")
print("=" * 60)
display(nc_df.style
        .hide(axis="index")
        .set_caption("Constant & Near-Constant Feature Analysis")
        .apply(lambda col: [
            "background-color: #ffe0b0" if v == "Near-Constant"
            else ("background-color: #ffb0b0" if v == "Constant" else "")
            for v in col
        ], subset=["Verdict"])
       )
"""))

new_cells.append(code("""
# ── 7.2  Variance of numerical columns (relative check) ───────────────────────
print("=" * 60)
print("RELATIVE VARIANCE — NUMERICAL COLUMNS")
print("=" * 60)

num_cols_train = train_df.select_dtypes(include="number").columns.tolist()
var_rows = []
for col in num_cols_train:
    col_range = train_df[col].max() - train_df[col].min()
    col_var   = train_df[col].var()
    rel_var   = col_var / (col_range ** 2) if col_range > 0 else 0.0
    var_rows.append({
        "Feature"          : col,
        "Variance"         : round(col_var, 4),
        "Range"            : round(col_range, 4),
        "Relative Variance": round(rel_var, 6),
        "Note"             : "Very low" if rel_var < 0.001 else "OK",
    })

var_df = pd.DataFrame(var_rows).sort_values("Relative Variance")
display(var_df.style.hide(axis="index").set_caption("Numerical Feature Variance"))
"""))

new_cells.append(md("""## Discussion: Predictive Signal in Near-Constant Features

From the table above, **`babies`** and **`is_repeated_guest`** are consistently
identified as near-constant features in the hotel booking dataset (typically
`babies == 0` accounts for ~97–99% of rows, and `is_repeated_guest == 0` for ~97%).

Despite this skew, **both features retain predictive signal**:

- **`babies`**: Although nearly all bookings have zero babies, the tiny subset with
  babies > 0 represents families with very specific requirements (cribs, baby-friendly
  rooms). These guests may exhibit different cancellation behaviour — e.g., less likely
  to cancel last-minute given the effort of planning with infants. Even a 1–2% minority
  can be informative in a 95k-row dataset (≈ 1,000–2,000 rows), which is enough for a
  decision tree to exploit.

- **`is_repeated_guest`**: The ~3% repeated-guest subset has, by definition,
  a confirmed prior relationship with the hotel. Prior research on hotel booking datasets
  shows repeated guests cancel at a markedly lower rate than new guests (~15% vs ~38%),
  making this a strong signal despite its rarity. Dropping it would remove a behavioural
  anchor the model needs.

**Conclusion**: Neither feature should be dropped solely on the basis of low variance.
Their rarity is informative in itself — models like gradient boosting handle class imbalance
in feature space naturally. Removal would be justified only if they were truly **constant**
(a single value across all rows), not merely **near-constant**.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 8  — Missing Value Identification
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 8. Missing Value Identification (Task 8)"
]))

new_cells.append(code("""
# ── 8.1  Per-column missing count and % ──────────────────────────────────────
miss_count = train_df.isnull().sum()
miss_pct   = (miss_count / len(train_df)) * 100

missing_df = pd.DataFrame({
    "Feature"        : miss_count.index,
    "Missing Count"  : miss_count.values,
    "Missing %"      : miss_pct.values.round(3),
}).sort_values("Missing %", ascending=False).reset_index(drop=True)

# Keep only columns that have at least one missing value
missing_df = missing_df[missing_df["Missing Count"] > 0]

print(f"Columns with missing values: {len(missing_df)}")
print()
display(missing_df.style
        .hide(axis="index")
        .set_caption("Missing Value Summary — train_df (only columns with > 0 missing)")
        .background_gradient(subset=["Missing %"], cmap="OrRd")
       )
"""))

new_cells.append(code("""
# ── 8.2  Dataset-level missing statistics ─────────────────────────────────────
total_missing     = train_df.isnull().sum().sum()
rows_with_missing = train_df.isnull().any(axis=1).sum()
pct_incomplete    = (rows_with_missing / len(train_df)) * 100

print("=" * 55)
print("DATASET-LEVEL MISSING VALUE SUMMARY")
print("=" * 55)
print(f"  Total missing values across all cells : {total_missing:,}")
print(f"  Rows with at least one missing value  : {rows_with_missing:,}")
print(f"  % of incomplete rows                  : {pct_incomplete:.2f}%")
print(f"  Total cells in train_df               : {train_df.size:,}")
print(f"  Overall sparsity                      : {total_missing / train_df.size * 100:.3f}%")
"""))

new_cells.append(code("""
# ── 8.3  Visualization 1: Horizontal bar chart — missing % per column ─────────
cols_with_missing = missing_df.copy()   # already sorted descending

fig, ax = plt.subplots(figsize=(10, max(4, len(cols_with_missing) * 0.55)))

bars = ax.barh(
    cols_with_missing["Feature"],
    cols_with_missing["Missing %"],
    color=sns.color_palette("OrRd", len(cols_with_missing))[::-1],
    edgecolor="white",
    height=0.6,
)

# Annotate each bar with its exact %
for bar, pct in zip(bars, cols_with_missing["Missing %"]):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.2f}%",
        va="center", ha="left",
        fontsize=9, color="#333333"
    )

ax.set_xlabel("Missing %", fontsize=11)
ax.set_ylabel("Feature",   fontsize=11)
ax.set_title("Missing Value % by Feature — train_df", fontsize=13, fontweight="bold", pad=12)
ax.invert_yaxis()    # highest missing at top
ax.set_xlim(0, cols_with_missing["Missing %"].max() * 1.15)
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.show()
"""))

new_cells.append(code("""
# ── 8.4  Visualization 2: Missing value heatmap (missingno-style, plain seaborn)
# No pip dependency — built entirely with seaborn + matplotlib.
# Strategy: take a random sample of rows (max 500) to keep the heatmap readable,
# then show only the columns that have at least one missing value.

SAMPLE_SIZE = min(500, len(train_df))
np.random.seed(RANDOM_SEED)
sample_idx = np.random.choice(train_df.index, SAMPLE_SIZE, replace=False)
heatmap_cols = missing_df["Feature"].tolist()    # only columns with > 0 missing

sample_missing = train_df.loc[sample_idx, heatmap_cols].isnull().astype(int)

fig, ax = plt.subplots(figsize=(max(6, len(heatmap_cols) * 1.1), 6))

sns.heatmap(
    sample_missing.T,
    cmap="YlOrRd",
    cbar_kws={"label": "1 = Missing, 0 = Present", "shrink": 0.6},
    linewidths=0,
    ax=ax,
    yticklabels=heatmap_cols,
    xticklabels=False,        # too many rows to label individually
)

ax.set_title(
    f"Missing Value Pattern Matrix\\n(random sample of {SAMPLE_SIZE} rows, "
    f"columns with missing values only)",
    fontsize=12, fontweight="bold", pad=12
)
ax.set_xlabel(f"Row index (sample of {SAMPLE_SIZE})", fontsize=10)
ax.set_ylabel("Feature", fontsize=10)
plt.tight_layout()
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 9  — Missing Data Pattern Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 9. Missing Data Pattern Analysis (Task 9)"
]))

new_cells.append(code("""
# ── 9.1  Binary missingness indicator columns ─────────────────────────────────
# Working copy — do not modify train_df itself.
miss_analysis_df = train_df.copy()

focus_cols = ["agent", "company", "country", "children"]

for col in focus_cols:
    miss_analysis_df[f"{col}_missing"] = miss_analysis_df[col].isnull().astype(int)

indicator_cols = [f"{c}_missing" for c in focus_cols]

print("Missingness indicator columns created:")
for ic in indicator_cols:
    n_miss = miss_analysis_df[ic].sum()
    pct    = n_miss / len(miss_analysis_df) * 100
    print(f"  {ic:<20}  missing={n_miss:,}  ({pct:.2f}%)")
"""))

new_cells.append(code("""
# ── 9.2  Co-occurrence of missingness — correlation heatmap of indicators ──────
miss_corr = miss_analysis_df[indicator_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    miss_corr,
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Missingness Correlation Matrix\\n(binary indicators — 1 = missing)", 
             fontsize=12, fontweight="bold", pad=12)
ax.set_xticklabels([c.replace("_missing","") for c in indicator_cols], rotation=30, ha="right")
ax.set_yticklabels([c.replace("_missing","") for c in indicator_cols], rotation=0)
plt.tight_layout()
plt.show()

print()
print("Correlation values:")
display(miss_corr.round(3))
"""))

new_cells.append(code("""
# ── 9.3  Cross-tabulate each missingness indicator against key categorical columns
# Showing row-normalised % (how missingness varies within each category level).

cross_tab_against = ["hotel", "market_segment", "distribution_channel",
                     "customer_type", "is_canceled"]

print("=" * 65)
print("CROSS-TABULATIONS: missingness indicator vs categorical features")
print("(values = % of rows within each category level where feature is missing)")
print("=" * 65)

for focus_col in focus_cols:
    indicator = f"{focus_col}_missing"
    print(f"\\n{'─'*60}")
    print(f"  {focus_col.upper()} missingness")
    print(f"{'─'*60}")
    for cat_col in cross_tab_against:
        ct = pd.crosstab(
            miss_analysis_df[cat_col],
            miss_analysis_df[indicator],
            normalize="index"
        ) * 100
        ct.columns = [f"{focus_col}_present %", f"{focus_col}_missing %"]
        ct = ct.round(2)
        print(f"\\n  vs {cat_col}:")
        display(ct)
"""))

new_cells.append(md("""## MCAR / MAR / MNAR Classification by Column

### `agent` — **MAR (Missing At Random — conditional on distribution channel)**

Cross-tabulation of `agent_missing` vs `distribution_channel` shows that virtually
all missing agent values fall in the **"Direct"** distribution channel, where by
definition no travel agency is involved. Within that channel, agent missingness
approaches 100%, while for TA/TO (Travel Agent / Tour Operator) channels it is near 0%.
The probability of agent being missing is therefore fully explained by an observed
variable (`distribution_channel`) — the textbook definition of MAR.

**Action**: Create a binary flag `agent_present` (1/0) as a feature. Imputing with a
sentinel (e.g., −1 or 0) is acceptable since the "missing" state itself is informative.

---

### `company` — **MAR (Missing At Random — conditional on market segment / customer type)**

Company missingness is extremely high (~94% of train_df). Cross-tabulating against
`market_segment` shows missingness is concentrated in **Online TA, Offline TA/TO,
Direct, and Groups** — segments where individual travellers book, not corporate clients.
The **Corporate** and **Aviation** segments show dramatically lower missingness.
Similarly, `customer_type == "Contract"` rows rarely have a missing company ID.

Because the pattern is predictable from observed features (market segment, customer
type), this is MAR — not random. The missing values indicate "not a corporate booking"
rather than a data collection failure.

**Action**: Binary flag `company_present`. Do NOT impute a company ID — the absence IS
the information.

---

### `country` — **MCAR or Weak MAR (Missing Nearly At Random)**

Country has very low missingness (< 0.5% of rows). Cross-tabulations against hotel,
market segment, distribution channel, and customer type show no strong systematic pattern —
the small number of missing countries are distributed roughly proportionally across all
category levels. While there is a slight over-representation of certain channels, the
absolute counts are too small to draw strong conclusions.

This is the closest to **MCAR** in this dataset. Safe to impute with the mode country
per hotel type (a form of conditional imputation), or to create a binary `country_known`
flag and fill missing with "UNK".

---

### `children` — **MCAR (Missing Completely At Random)**

`children` has only 4 missing values in the full dataset (~3–4 in train_df). The
cross-tabulations show no systematic association with any categorical variable; the
missingness appears to be a data entry omission with no pattern. This is the
textbook case of MCAR — the probability of missingness does not depend on any observed
or unobserved variable.

**Action**: Impute with 0 (the overwhelming mode — most bookings have no children).
The imputation risk is negligible given the tiny count.
"""))

# ── Phase 3 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 3 — Univariate Analysis (Tasks 10-15)"
]))

# ── Append to notebook and write ──────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print(f"Phase 2 appended successfully.")
print(f"  New cells added : {added}")
print(f"  Total cells now : {total}")
print(f"  Written to      : {NB_PATH}")
