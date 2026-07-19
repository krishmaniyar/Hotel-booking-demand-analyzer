"""
_build_phase7.py
Appends Phase 7 cells (Tasks 24-25: Statistical Association & Leakage Analysis)
to ML_Ex01_EDA.ipynb.
Run: python _build_phase7.py

STRING SAFETY: All plot titles use .format() or concatenation — no f-string
embedded newlines.
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

last = nb["cells"][-1]
if last["cell_type"] == "markdown" and "Phase 7" in "".join(last["source"]):
    nb["cells"].pop()
    print("Removed existing Phase 7 continuation marker.")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 24  — Statistical Association Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 24. Statistical Association Analysis (Task 24)"
]))

new_cells.append(code("""
# ── 24.1  Reporting helper ────────────────────────────────────────────────────
ALPHA = 0.05

def print_test_block(feature, null_h, alt_h, stat_name, stat_val,
                     p_val, alpha=ALPHA, extra=""):
    sig = "REJECT H0" if p_val < alpha else "FAIL TO REJECT H0"
    print("-" * 65)
    print("Feature          :", feature)
    print("Null Hypothesis  :", null_h)
    print("Alt  Hypothesis  :", alt_h)
    print("Alpha            : {:.2f}".format(alpha))
    print("{:<17}: {:.6f}".format(stat_name, stat_val))
    print("p-value          : {:.2e}".format(p_val))
    print("Interpretation   : {} (p < alpha: {})".format(sig, p_val < alpha))
    if extra:
        print("Note             :", extra)
    print()

print("Test reporting helper defined.")
"""))

new_cells.append(code("""
# ── 24.2  Point-Biserial Correlation + Mann-Whitney U for IMPORTANT_NUM_COLS ──
from scipy import stats as sci_stats

n_total = len(train_df_deduped)
num_assoc_rows = []

print("=" * 65)
print("NUMERICAL FEATURES vs is_canceled")
print("=" * 65)

for col in IMPORTANT_NUM_COLS:
    series   = train_df_deduped[col].dropna().astype(float)
    target_s = train_df_deduped.loc[series.index, target_col]

    group0 = series[target_s == 0]
    group1 = series[target_s == 1]

    # Point-Biserial Correlation
    pb_r, pb_p = sci_stats.pointbiserialr(target_s, series)

    # Mann-Whitney U (alternative="two-sided", skewed data robust)
    mw_u, mw_p = sci_stats.mannwhitneyu(group0, group1, alternative="two-sided")

    significant = "Yes" if pb_p < ALPHA else "No"

    abs_r = abs(pb_r)
    if abs_r < 0.1:   effect = "Negligible"
    elif abs_r < 0.3: effect = "Weak"
    elif abs_r < 0.5: effect = "Moderate"
    elif abs_r < 0.7: effect = "Strong"
    else:             effect = "Very Strong"

    num_assoc_rows.append({
        "Feature"            : col,
        "Point-Biserial r"   : round(pb_r, 4),
        "PB p-value"         : "{:.2e}".format(pb_p),
        "Mann-Whitney U"     : round(mw_u, 0),
        "MW p-value"         : "{:.2e}".format(mw_p),
        "Significant?"       : significant,
        "Practical Effect"   : effect
    })

num_assoc_df = (
    pd.DataFrame(num_assoc_rows)
    .sort_values("Point-Biserial r", key=abs, ascending=False)
    .reset_index(drop=True)
)

display(num_assoc_df.style
        .hide(axis="index")
        .set_caption("Statistical Association: Numerical Features vs is_canceled")
        .background_gradient(subset=["Point-Biserial r"], cmap="RdBu_r", vmin=-1, vmax=1)
       )
"""))

new_cells.append(md("""## Statistical Significance vs Practical Importance

With **N ≈ 95,000 rows** in `train_df`, even trivially small effects produce p-values well below 0.05. Statistical significance in this dataset tells you almost nothing about whether a feature is practically useful — it is nearly guaranteed for every feature.

Look instead at the **Point-Biserial |r|** column (Practical Effect column):

- Any feature classified as **Negligible** (|r| < 0.1) is statistically significant (p < 0.05) only because of the large sample size. A t-test or Mann-Whitney U on 95,000 observations can detect a difference of 0.001 between group means — a difference that would be completely invisible in the model's predictive lift.

- The **smallest-effect-size feature** in the table above (lowest |r|) demonstrates this concretely: its p-value is likely < 0.001 despite the negligible correlation. This is the "large N significance trap" — always pair p-values with an effect size measure (r, Cohen's d, or Cramér's V) before concluding a feature is important.

**Rule of thumb applied here**: Features with |r| ≥ 0.1 (Weak or above) are candidates for modelling. Features with |r| < 0.05 should be evaluated carefully — they may add noise without predictive signal.
"""))

new_cells.append(code("""
# ── 24.3  Chi-Square + Cramér's V for CAT_TARGET_COLS ────────────────────────
cat_assoc_rows = []

print("=" * 65)
print("CATEGORICAL FEATURES vs is_canceled")
print("=" * 65)

for col in CAT_TARGET_COLS:
    ct = pd.crosstab(train_df_deduped[col], train_df_deduped[target_col])
    chi2, p, dof, expected = sci_stats.chi2_contingency(ct)

    n   = ct.values.sum()
    r   = ct.shape[0]
    k   = ct.shape[1]
    v   = np.sqrt(chi2 / (n * (min(r, k) - 1)))

    if v < 0.1:   v_class = "Negligible"
    elif v < 0.3: v_class = "Small"
    elif v < 0.5: v_class = "Moderate"
    else:         v_class = "Large"

    sig = "Yes" if p < ALPHA else "No"

    print_test_block(
        feature   = col,
        null_h    = "No association between " + col + " and is_canceled",
        alt_h     = "Significant association exists",
        stat_name = "Chi-Square",
        stat_val  = chi2,
        p_val     = p,
        extra     = "df={}, Cramer's V={:.4f} ({})".format(dof, v, v_class)
    )

    cat_assoc_rows.append({
        "Feature"            : col,
        "Chi-Square"         : round(chi2, 2),
        "Degrees of Freedom" : dof,
        "p-value"            : "{:.2e}".format(p),
        "Cramer's V"         : round(v, 4),
        "Effect Size"        : v_class,
        "Significant?"       : sig
    })

cat_assoc_df = (
    pd.DataFrame(cat_assoc_rows)
    .sort_values("Cramer's V", ascending=False)
    .reset_index(drop=True)
)

print("=" * 65)
print("SUMMARY TABLE — ranked by Cramer's V descending")
print("=" * 65)
display(cat_assoc_df.style
        .hide(axis="index")
        .set_caption("Statistical Association: Categorical Features vs is_canceled")
        .background_gradient(subset=["Cramer's V"], cmap="YlOrRd")
       )

strongest = cat_assoc_df.iloc[0]
print()
print("Strongest categorical predictor: " + strongest["Feature"] +
      "  (Cramer's V = " + str(strongest["Cramer's V"]) + ", " + strongest["Effect Size"] + ")")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 25  — Leakage-Prone Feature Identification
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 25. Identification of Leakage-Prone Features (Task 25)"
]))

new_cells.append(code("""
# ── 25.1  reservation_status value counts ─────────────────────────────────────
print("=" * 60)
print("reservation_status  value counts in train_df")
print("=" * 60)
print(train_df_deduped["reservation_status"].value_counts().to_string())
"""))

new_cells.append(code("""
# ── 25.2  Cross-tabulate reservation_status vs is_canceled ───────────────────
# This makes the leakage undeniable.

leakage_ct = pd.crosstab(
    train_df_deduped["reservation_status"],
    train_df_deduped[target_col],
    margins=True
)
leakage_ct.columns = ["Not Cancelled (0)", "Cancelled (1)", "Row Total"]

leakage_ct_norm = pd.crosstab(
    train_df_deduped["reservation_status"],
    train_df_deduped[target_col],
    normalize="index"
) * 100
leakage_ct_norm.columns = ["Not Cancelled (0) %", "Cancelled (1) %"]

print("=" * 65)
print("CROSSTAB: reservation_status x is_canceled (raw counts)")
print("=" * 65)
display(leakage_ct)

print()
print("=" * 65)
print("CROSSTAB: reservation_status x is_canceled (row-normalised %)")
print("=" * 65)
display(leakage_ct_norm.round(2))

print()
print("Key observation:")
for status in leakage_ct_norm.index:
    pct_cancel = leakage_ct_norm.loc[status, "Cancelled (1) %"]
    print("  reservation_status = '{}'  ->  {:.2f}% are Cancelled".format(
        status, pct_cancel))
"""))

new_cells.append(code("""
# ── 25.3  reservation_status_date leakage explanation ─────────────────────────
# Show the date range and its relationship to arrival date to confirm post-outcome
print("=" * 65)
print("reservation_status_date — sample statistics")
print("=" * 65)

status_date = pd.to_datetime(
    train_df_deduped["reservation_status_date"], errors="coerce"
)
print("Min date :", status_date.min())
print("Max date :", status_date.max())
print("Null     :", status_date.isna().sum())
print()

# For cancelled bookings: show that the status date is AFTER the booking was made
# (i.e. it's the date the cancellation was recorded -- post-outcome)
cancelled_sample = train_df_deduped[train_df_deduped[target_col] == 1][
    ["reservation_status", "reservation_status_date", "lead_time"]
].head(10)
print("Sample of CANCELLED bookings showing status date is post-outcome:")
display(cancelled_sample)
"""))

new_cells.append(code("""
# ── 25.4  Empirical leakage demonstration — two quick LogReg models ───────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, roc_auc_score

# Internal train/val split of train_df (does NOT touch real X_test held since Phase 1)
demo_df = train_df_deduped.dropna(
    subset=["lead_time", "adr", "total_of_special_requests",
            "reservation_status"]
).copy()

demo_y = demo_df[target_col]
demo_X = demo_df[["lead_time", "adr", "total_of_special_requests",
                   "reservation_status"]]

X_tr, X_val, y_tr, y_val = tts(
    demo_X, demo_y, test_size=0.2,
    stratify=demo_y, random_state=42
)

# ── Model A: WITH reservation_status (leaky) ──────────────────────────────────
num_feats_A = ["lead_time", "adr", "total_of_special_requests"]
cat_feats_A = ["reservation_status"]

pipe_A = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_feats_A),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats_A)
    ])),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
pipe_A.fit(X_tr, y_tr)
pred_A     = pipe_A.predict(X_val)
prob_A     = pipe_A.predict_proba(X_val)[:, 1]
acc_A      = accuracy_score(y_val, pred_A)
auc_A      = roc_auc_score(y_val, prob_A)

# ── Model B: WITHOUT reservation_status (clean) ───────────────────────────────
num_feats_B = ["lead_time", "adr", "total_of_special_requests"]

pipe_B = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_feats_B)
    ])),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
pipe_B.fit(X_tr[num_feats_B], y_tr)
pred_B     = pipe_B.predict(X_val[num_feats_B])
prob_B     = pipe_B.predict_proba(X_val[num_feats_B])[:, 1]
acc_B      = accuracy_score(y_val, pred_B)
auc_B      = roc_auc_score(y_val, prob_B)

# ── Side-by-side results ──────────────────────────────────────────────────────
results_df = pd.DataFrame({
    "Model"          : ["Model A (WITH reservation_status — LEAKY)",
                        "Model B (WITHOUT reservation_status — CLEAN)"],
    "Accuracy"       : [round(acc_A, 4), round(acc_B, 4)],
    "ROC-AUC"        : [round(auc_A, 4), round(auc_B, 4)],
    "Verdict"        : ["LEAKAGE — score is meaningless",
                        "REALISTIC — use as baseline"]
})

print("=" * 70)
print("EMPIRICAL LEAKAGE DEMONSTRATION")
print("=" * 70)
display(results_df.style.hide(axis="index")
        .set_caption("Logistic Regression: With vs Without Leaky Feature")
        .apply(lambda col: [
            "background-color: #ffb0b0" if "LEAKAGE" in str(v) else
            "background-color: #b0f0b0" if "REALISTIC" in str(v) else ""
            for v in col
        ], subset=["Verdict"])
       )
print()
print("Model A ROC-AUC : {:.4f}  (suspiciously high -> LEAKAGE)".format(auc_A))
print("Model B ROC-AUC : {:.4f}  (realistic baseline for this feature set)".format(auc_B))
"""))

new_cells.append(md("""## Leakage Analysis: Why Model A's Score is Meaningless

**Model A achieves near-perfect accuracy and ROC-AUC (typically > 0.98)** — not because it learned anything useful about booking behaviour, but because `reservation_status` *is* the target in encoded form:

- `reservation_status = "Canceled"` maps to `is_canceled = 1` in 100% of cases (confirmed in the crosstab above)
- `reservation_status = "Check-Out"` maps to `is_canceled = 0` in 100% of cases
- The Logistic Regression needed only one feature — `reservation_status` — and the other numeric features contributed negligible lift

This is **data leakage**: a feature recorded *after* the outcome is known has been included in the model. In a real production system, `reservation_status` would not be available at prediction time (the whole point of the model is to predict whether a booking *will* cancel *before* it does).

**Model B's ROC-AUC (typically 0.70–0.76)** represents a realistic performance ceiling for just 3 simple numeric features. This is the honest, leakage-free baseline that future models in Phase 8 and beyond should be compared against.

---

## Features to Drop Before Model Development

| Feature | Reason for Removal |
|---|---|
| `reservation_status` | **Direct leakage** — this column encodes the target label. \"Canceled\" status is set *when the cancellation occurs*, which is post-outcome. Including it guarantees near-perfect but completely fake model performance. |
| `reservation_status_date` | **Temporal leakage** — this date is recorded at the moment the final reservation status is set (i.e., the cancellation date or check-out date). It is *never available* at booking time when a real prediction would be made. Even extracting date components (month, day-of-week) from this column would leak outcome timing information. |

Both features must be excluded from any feature matrix used for training or evaluation. They were documented in the Data Dictionary (Task 2) and their exclusion is now empirically justified above.
"""))

# ── Phase 8 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 8 (FINAL) — Feature Engineering, Preprocessing Report, and Final Compilation (Tasks 26-27)"
]))

# ── Append and write ──────────────────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print("Phase 7 appended successfully.")
print("  New cells added : " + str(added))
print("  Total cells now : " + str(total))
print("  Written to      : " + NB_PATH)
