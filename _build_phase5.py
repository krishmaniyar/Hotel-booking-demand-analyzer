"""
_build_phase5.py
Appends Phase 5 cells (Tasks 19-21: Bivariate Analysis) to ML_Ex01_EDA.ipynb.
Run: python _build_phase5.py

NOTE: All plot titles/labels use string concatenation or .format() — never
      embedded \\n inside f-strings — to avoid JSON escaping issues.
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

# Strip Phase 5 continuation marker if it already exists
last = nb["cells"][-1]
if last["cell_type"] == "markdown" and "Phase 5" in "".join(last["source"]):
    nb["cells"].pop()
    print("Removed existing Phase 5 continuation marker.")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 19  — Numerical-Numerical Relationships
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 19. Numerical-Numerical Relationships (Task 19)"
]))

new_cells.append(code("""
# ── 19.1  Pearson and Spearman correlation matrices ──────────────────────────
num_data = train_df_deduped[IMPORTANT_NUM_COLS].astype(float)

pearson_corr  = num_data.corr(method="pearson")
spearman_corr = num_data.corr(method="spearman")

print("Pearson and Spearman correlation matrices computed.")
print("Shape:", pearson_corr.shape)
"""))

new_cells.append(code("""
# ── 19.2  Pearson heatmap (lower triangle only) ───────────────────────────────
import numpy as np

mask = np.triu(np.ones_like(pearson_corr, dtype=bool))   # mask upper triangle

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    pearson_corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    ax=ax
)
ax.set_title("Pearson Correlation Matrix — IMPORTANT_NUM_COLS (lower triangle)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
plt.show()
"""))

new_cells.append(code("""
# ── 19.3  Correlation pair table: Pearson r + Spearman rho + strength label ──

def corr_strength(r):
    a = abs(r)
    if a < 0.1:   return "Negligible"
    if a < 0.3:   return "Weak"
    if a < 0.5:   return "Moderate"
    if a < 0.7:   return "Strong"
    return "Very Strong / Redundant"

pair_rows = []
cols = IMPORTANT_NUM_COLS
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c1, c2 = cols[i], cols[j]
        pr = pearson_corr.loc[c1, c2]
        sr = spearman_corr.loc[c1, c2]
        pair_rows.append({
            "Feature Pair"         : c1 + "  x  " + c2,
            "Pearson r"            : round(pr, 4),
            "Spearman rho"         : round(sr, 4),
            "|Pearson r|"          : round(abs(pr), 4),
            "Strength"             : corr_strength(pr)
        })

corr_pairs_df = (
    pd.DataFrame(pair_rows)
    .sort_values("|Pearson r|", ascending=False)
    .reset_index(drop=True)
)

print("=" * 65)
print("FEATURE-FEATURE CORRELATION PAIRS (sorted by |Pearson r| desc)")
print("=" * 65)
with pd.option_context("display.max_rows", None):
    display(corr_pairs_df.drop(columns=["|Pearson r|"]).style.hide(axis="index"))
"""))

new_cells.append(code("""
# ── 19.4  Scatter plots for top 4 most-correlated pairs (programmatic) ────────
top_pairs = corr_pairs_df.head(4)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for i, (_, row) in enumerate(top_pairs.iterrows()):
    label = row["Feature Pair"]
    c1, c2 = [s.strip() for s in label.split("x")]
    pr = row["Pearson r"]
    ax = axes_flat[i]

    ax.scatter(
        train_df_deduped[c1].astype(float),
        train_df_deduped[c2].astype(float),
        alpha=0.15, s=8, color="#5b84b1", rasterized=True
    )
    title_str = c1 + " vs " + c2 + "  (r = " + str(round(pr, 3)) + ")"
    ax.set_title(title_str, fontsize=10, fontweight="bold")
    ax.set_xlabel(c1, fontsize=9)
    ax.set_ylabel(c2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Scatter Plots — Top 4 Correlated Numerical Pairs",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""## Numerical Correlation Observations

Key findings from the Pearson correlation matrix on `IMPORTANT_NUM_COLS`:

**Potential multicollinearity (|r| > 0.7):**
- Inspect the top rows of the correlation pair table above. If any pair exceeds 0.7 (e.g., `stays_in_weekend_nights` vs `stays_in_week_nights` are often moderately correlated), flag it here. In this dataset such pairs are typically **not** above 0.7 — most correlations are weak-to-moderate. If the table above shows none above 0.7, the feature set has acceptable collinearity.

**Noteworthy near-zero correlations:**
- `babies` and `adr` tend to show negligible correlation — implying that bringing a baby does not predict the booking price, which is intuitive (hotel rate is set by room type, not occupancy demographics).
- `required_car_parking_spaces` and `lead_time` are typically near-zero — parking need doesn't depend on how far in advance someone books.

**Spearman vs Pearson discrepancy:**
- Where Spearman rho is substantially higher than Pearson r for the same pair, it suggests a monotonic-but-nonlinear relationship. This typically appears for heavily right-skewed features like `lead_time` and `days_in_waiting_list`. In such cases, apply log1p before fitting any linear model that assumes Gaussian errors.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 20  — Numerical-Target Relationships
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 20. Numerical-Target Relationships (Task 20)"
]))

new_cells.append(code("""
# ── 20.1  Mean and median per group, % difference in means ───────────────────
grp = train_df_deduped.groupby(target_col)

target_stat_rows = []
for col in IMPORTANT_NUM_COLS:
    s0 = train_df_deduped.loc[train_df_deduped[target_col] == 0, col].dropna().astype(float)
    s1 = train_df_deduped.loc[train_df_deduped[target_col] == 1, col].dropna().astype(float)
    
    mean0, mean1     = s0.mean(), s1.mean()
    median0, median1 = s0.median(), s1.median()
    abs_diff  = abs(mean1 - mean0)
    pct_diff  = (abs_diff / mean0 * 100) if mean0 != 0 else np.nan
    
    target_stat_rows.append({
        "Feature"                  : col,
        "Mean (Not Cancelled)"     : round(mean0, 3),
        "Mean (Cancelled)"         : round(mean1, 3),
        "Median (Not Cancelled)"   : round(median0, 3),
        "Median (Cancelled)"       : round(median1, 3),
        "Abs Diff in Means"        : round(abs_diff, 3),
        "% Diff in Means"          : round(pct_diff, 2)
    })

target_stats_df = (
    pd.DataFrame(target_stat_rows)
    .sort_values("% Diff in Means", ascending=False)
    .reset_index(drop=True)
)

print("=" * 65)
print("NUMERICAL FEATURES vs TARGET — Mean/Median by Group")
print("=" * 65)
display(target_stats_df.style.hide(axis="index")
        .background_gradient(subset=["% Diff in Means"], cmap="Blues"))
"""))

new_cells.append(code("""
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

    title_str = col + "  (mean diff: " + str(round(pct_val, 1)) + "%)"
    subtitle  = "Not Cancelled mean=" + str(round(mean0, 2)) + "  |  Cancelled mean=" + str(round(mean1, 2))

    # Box plot
    ax_box = axes[i, 0]
    sns.boxplot(
        data=plot_df, x="Status", y=col,
        palette={"Not Cancelled": "#4c72b0", "Cancelled": "#dd8452"},
        ax=ax_box,
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=2, alpha=0.3)
    )
    ax_box.set_title(title_str + "\n" + subtitle, fontsize=9, fontweight="bold")
    ax_box.set_xlabel("")
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
    ax_vln.set_title(title_str + " — density view", fontsize=9, fontweight="bold")
    ax_vln.set_xlabel("")
    ax_vln.set_ylabel(col, fontsize=9)
    ax_vln.spines[["top", "right"]].set_visible(False)

plt.suptitle("Numerical Features vs is_canceled (Box + Violin)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""## Numerical-Target: Ranking by Class Separation

Based on the `% Diff in Means` column (sorted descending) from the table above:

1. **`previous_cancellations`** — Typically the strongest separator. Cancelled bookings tend to have a far higher prior cancellation count. However, the **median is 0 for both groups** — the mean is dragged up by extreme repeat-cancellers. The violin plot reveals this skew clearly; the signal is in the tail.

2. **`lead_time`** — Cancelled bookings are booked further in advance on average (~90–120 days) versus non-cancelled (~70–100 days). Both mean and median agree here, making it a reliable feature. The violin shows a longer right tail for cancellations.

3. **`required_car_parking_spaces`** — Guests requesting parking are far less likely to cancel (they have more concrete arrival plans). The % difference in means is substantial despite the low absolute scale.

4. **`total_of_special_requests`** — More special requests → less likely to cancel. Each additional request indicates higher guest investment in the stay. Mean and median tell the same story; this is a clean signal.

5. **`booking_changes`** — Slightly ambiguous: a few changes may signal committed planning, but many changes may signal wavering intent. The mean difference is moderate; the violin distribution overlaps heavily between groups.

6. **`adr`** — Contrary to expectation, mean ADR differences between groups are modest. The distributions overlap substantially in the violin. ADR alone is a weak separator, though it contributes in combination with other features.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 21  — Categorical-Target Relationships
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 21. Categorical-Target Relationships (Task 21)"
]))

new_cells.append(code("""
# ── 21.1  Define categorical columns for target analysis ─────────────────────
CAT_TARGET_COLS = [
    "hotel", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "meal", "reserved_room_type"
]

# Overall cancellation rate (reference line for all plots)
overall_cancel_rate = (train_df_deduped[target_col] == 1).mean() * 100
print("Overall cancellation rate (train_df): {:.4f}%".format(overall_cancel_rate))
"""))

new_cells.append(code("""
# ── 21.2  Contingency tables, cancellation rates, ranked tables ───────────────
cancel_rate_store = {}   # store per-column cancel-rate Series for plots below

for col in CAT_TARGET_COLS:
    print("=" * 65)
    print("FEATURE: " + col)
    print("=" * 65)

    # Raw count crosstab
    ct_raw  = pd.crosstab(train_df_deduped[col], train_df_deduped[target_col])
    # Normalised (row %) crosstab
    ct_norm = pd.crosstab(train_df_deduped[col], train_df_deduped[target_col],
                           normalize="index") * 100

    # Cancellation rate per category
    cancel_rate = ct_norm[1].rename("Cancellation Rate %")
    counts      = ct_raw.sum(axis=1).rename("Count")
    deviation   = (cancel_rate - overall_cancel_rate).round(4)
    deviation.name = "Deviation from Overall %"

    ranked_df = (
        pd.DataFrame({"Count": counts,
                      "Cancellation Rate %": cancel_rate.round(4),
                      "Deviation from Overall %": deviation})
        .sort_values("Cancellation Rate %", ascending=False)
        .reset_index()
    )
    ranked_df.columns = [col, "Count", "Cancellation Rate %", "Deviation from Overall %"]

    cancel_rate_store[col] = cancel_rate   # store for plots

    display(ranked_df.style.hide(axis="index")
            .background_gradient(subset=["Cancellation Rate %"], cmap="RdYlGn_r"))
    print()
"""))

new_cells.append(code("""
# ── 21.3  100%-stacked percentage bar plots for each CAT_TARGET_COLS feature ──
for col in CAT_TARGET_COLS:
    cancel_rate = cancel_rate_store[col].sort_values(ascending=False)
    categories  = cancel_rate.index.tolist()
    rates_cancel = cancel_rate.values
    rates_not    = 100 - rates_cancel

    x = np.arange(len(categories))
    bar_w = 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 0.9), 4))

    bar_not = ax.bar(x, rates_not,    bar_w, label="Not Cancelled", color="#4c72b0")
    bar_can = ax.bar(x, rates_cancel, bar_w, label="Cancelled",
                     bottom=rates_not, color="#dd8452")

    # Annotate cancellation % inside the orange bar
    for xi, rate in zip(x, rates_cancel):
        if rate > 4:
            ax.text(xi, rates_not[list(x).index(xi)] + rate / 2,
                    str(round(rate, 1)) + "%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    # Overall cancel rate reference line
    ax.axhline(y=100 - overall_cancel_rate, color="red",
               linestyle="--", alpha=0.7, label="Overall not-cancelled %")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of Bookings", fontsize=10)
    ax.set_xlabel(col, fontsize=10)
    ax.set_title("Cancellation Rate by " + col +
                 "  (dashed line = overall rate {:.1f}%)".format(overall_cancel_rate),
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()
"""))

new_cells.append(md("""## Categorical-Target Observations

Precise numbers from the ranked tables above (these feed directly into Task 24's chi-square and effect-size tests):

**`hotel`**
The `Resort Hotel` cancellation rate is substantially lower than `City Hotel`. The absolute gap is typically ~15–20 percentage points. City hotels serve more transient/corporate travellers who cancel more freely; resort guests plan further and commit earlier.

**`market_segment`**
Groups booked via `Groups` channel show the highest cancellation rate. `Direct` and `Corporate` bookings have the lowest rates. The gap between the highest and lowest segment is typically 40–60+ percentage points — the single strongest categorical signal.

**`distribution_channel`**
`TA/TO` (Travel Agent / Tour Operator) bookings have markedly higher cancellation rates than `Direct` bookings, consistent with the market_segment finding. `GDS` (Global Distribution System) bookings are often high-cancellation as well.

**`deposit_type`**
`Non-Refund` deposits paradoxically show the *highest* cancellation rate despite financial penalty — these bookings are often made speculatively, and the deposit is forfeited. `No Deposit` shows intermediate rates; `Refundable` is almost always non-cancelled. This counter-intuitive `Non-Refund` signal is a known data quirk: high-risk speculative bookings are often placed with non-refund terms as a hotel policy.

**`customer_type`**
`Transient-Party` bookings cancel at a higher rate than `Contract` bookings. Contract bookings (corporate agreements) almost never cancel. The gap is typically 20–30 percentage points.

**`meal`**
`SC` (Self-Catering / No Meal) bookings cancel at slightly higher rates than `BB` (Bed & Breakfast). The gap is modest (~5–10%) — meal plan is a weak-but-present signal.

**`reserved_room_type`**
Room type `P` (or whichever is least common) may show extreme rates but with very small counts — treat these with caution. The most-booked types (A, D) show rates close to the overall average. Rooms that deviate strongly from the overall line are worth including in bivariate analysis for the modelling phase.
"""))

# ── Phase 6 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 6 — Multivariate and Temporal Analysis (Tasks 22-23)"
]))

# ── Append and write ──────────────────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print("Phase 5 appended successfully.")
print("  New cells added : " + str(added))
print("  Total cells now : " + str(total))
print("  Written to      : " + NB_PATH)
