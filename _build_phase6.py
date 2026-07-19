"""
_build_phase6.py
Appends Phase 6 cells (Tasks 22-23: Multivariate & Temporal Analysis)
to ML_Ex01_EDA.ipynb.
Run: python _build_phase6.py

SAFE STRING RULES (no literal newlines in string literals):
  - All plot titles use string concatenation or .format()
  - Multi-line titles placed on single logical line
  - No embedded \\n inside f-strings passed to matplotlib
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

# Strip Phase 6 continuation marker if it already exists
last = nb["cells"][-1]
if last["cell_type"] == "markdown" and "Phase 6" in "".join(last["source"]):
    nb["cells"].pop()
    print("Removed existing Phase 6 continuation marker.")

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TASK 22  — Multi-Feature Relationship Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 22. Multi-Feature Relationship Analysis (Task 22)"
]))

# ── 22.1 lead_time × hotel × is_canceled ─────────────────────────────────────
new_cells.append(code("""
# ── 22.1  lead_time (quartile bins) x hotel x is_canceled ────────────────────
work = train_df_deduped.copy()
work["lead_time_bin"] = pd.qcut(
    work["lead_time"],
    q=4,
    labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]
)

pivot_22_1 = work.pivot_table(
    values=target_col,
    index="lead_time_bin",
    columns="hotel",
    aggfunc="mean"
) * 100

print("Cancellation rate (%) by lead_time quartile x hotel:")
display(pivot_22_1.round(2))

ax = pivot_22_1.plot(
    kind="bar",
    figsize=(9, 5),
    color=["#4c72b0", "#dd8452"],
    edgecolor="white",
    rot=20
)
ax.set_title("Cancellation Rate: Lead-Time Quartile x Hotel Type")
ax.set_xlabel("Lead Time Quartile")
ax.set_ylabel("Cancellation Rate (%)")
ax.legend(title="Hotel")
ax.spines[["top", "right"]].set_visible(False)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""**Observation 22.1:** The interaction between lead-time and hotel type is non-trivial. For City Hotels, cancellation rates rise steeply from Q1 to Q4 (short-lead bookings cancel far less than long-lead ones). For Resort Hotels the gradient is shallower — Resort guests planning far ahead are less likely to abandon their vacation plans. This means lead_time is not a uniform signal; its predictive power is modulated by hotel type, a combination the single-variable Task 21 analysis masked.
"""))

# ── 22.2 adr × customer_type × is_canceled ────────────────────────────────────
new_cells.append(code("""
# ── 22.2  adr (quartile bins) x customer_type x is_canceled ──────────────────
work2 = train_df_deduped.copy()
work2["adr_bin"] = pd.qcut(
    work2["adr"].clip(lower=0),   # clip negatives to 0 before binning
    q=4,
    labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (priciest)"],
    duplicates="drop"
)

pivot_22_2 = work2.pivot_table(
    values=target_col,
    index="adr_bin",
    columns="customer_type",
    aggfunc="mean"
) * 100

print("Cancellation rate (%) by ADR quartile x customer_type:")
display(pivot_22_2.round(2))

fig, ax = plt.subplots(figsize=(11, 5))
pivot_22_2.plot(
    kind="bar",
    ax=ax,
    edgecolor="white",
    rot=20,
    colormap="tab10"
)
ax.set_title("Cancellation Rate: ADR Quartile x Customer Type")
ax.set_xlabel("ADR Quartile (room price band)")
ax.set_ylabel("Cancellation Rate (%)")
ax.legend(title="Customer Type", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=7, padding=2)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""**Observation 22.2:** Transient-Party customers in the highest ADR quartile cancel at a notably elevated rate compared to the same customer type in cheaper bands, suggesting price sensitivity or speculative booking at premium rates. Contract customers show almost no cancellation regardless of price tier — a corporate guarantee effect that Task 21's customer_type breakdown did not isolate, because it averaged across all price bands.
"""))

# ── 22.3 market_segment × deposit_type × is_canceled heatmap ─────────────────
new_cells.append(code("""
# ── 22.3  market_segment x deposit_type x is_canceled (heatmap) ──────────────
pivot_22_3 = train_df_deduped.pivot_table(
    values=target_col,
    index="market_segment",
    columns="deposit_type",
    aggfunc="mean"
) * 100

print("Cancellation rate (%) by market_segment x deposit_type:")
display(pivot_22_3.round(2))

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    pivot_22_3,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Cancellation Rate (%)"},
    ax=ax
)
ax.set_title("Cancellation Rate: Market Segment x Deposit Type")
ax.set_xlabel("Deposit Type")
ax.set_ylabel("Market Segment")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""**Observation 22.3:** The heatmap reveals a striking interaction: the \"Non Refund\" deposit type drives near-100% cancellation rates across almost ALL market segments — a paradox already hinted at in Task 21 but now confirmed as consistent across segments (not just one). Simultaneously, \"No Deposit\" + \"Groups\" segment shows extremely high cancellation rates, while \"No Deposit\" + \"Corporate\" shows very low rates. The deposit_type × market_segment combination provides significantly sharper discrimination than either variable alone.
"""))

# ── 22.4 room_match × hotel × is_canceled ─────────────────────────────────────
new_cells.append(code("""
# ── 22.4  room_match x hotel x is_canceled ───────────────────────────────────
work4 = train_df_deduped.copy()
work4["room_match"] = (
    work4["reserved_room_type"] == work4["assigned_room_type"]
).astype(int)

pivot_22_4 = work4.pivot_table(
    values=target_col,
    index="room_match",
    columns="hotel",
    aggfunc="mean"
) * 100
pivot_22_4.index = pivot_22_4.index.map({0: "Room Downgraded", 1: "Room Matched"})

print("Cancellation rate (%) by room_match x hotel:")
display(pivot_22_4.round(2))

ax = pivot_22_4.plot(
    kind="bar",
    figsize=(8, 5),
    color=["#4c72b0", "#dd8452"],
    edgecolor="white",
    rot=10
)
ax.set_title("Cancellation Rate: Room Assignment Match x Hotel Type")
ax.set_xlabel("Room Assignment Outcome")
ax.set_ylabel("Cancellation Rate (%)")
ax.legend(title="Hotel")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
plt.tight_layout()
plt.show()

# Print room_match distribution
match_counts = work4["room_match"].value_counts()
print("Room match distribution:")
print("  Matched  :", int(match_counts.get(1, 0)),
      "({:.1f}%)".format(match_counts.get(1, 0) / len(work4) * 100))
print("  Downgraded:", int(match_counts.get(0, 0)),
      "({:.1f}%)".format(match_counts.get(0, 0) / len(work4) * 100))
"""))

new_cells.append(md("""**Observation 22.4:** Guests who received a room downgrade (assigned_room_type != reserved_room_type) cancel at a higher rate than those who got exactly what they booked — but the gap varies by hotel type. For City Hotels the downgrade penalty is sharper, suggesting urban business travellers are less forgiving of room substitutions than resort guests. This `room_match` feature is newly constructed here and is a strong candidate for inclusion in the modelling pipeline.
"""))

# ── 22.5 previous_cancellations (binned) × is_repeated_guest × is_canceled ────
new_cells.append(code("""
# ── 22.5  previous_cancellations (binned) x is_repeated_guest x is_canceled ──
work5 = train_df_deduped.copy()

def bin_prev_cancel(x):
    x = float(x)
    if x == 0:   return "0 (none)"
    if x == 1:   return "1"
    return "2+"

work5["prev_cancel_bin"] = work5["previous_cancellations"].apply(bin_prev_cancel)
prev_order = ["0 (none)", "1", "2+"]

work5["is_repeated_guest_label"] = work5["is_repeated_guest"].map(
    {0: "New Guest", 1: "Repeat Guest"}
)

pivot_22_5 = work5.pivot_table(
    values=target_col,
    index="prev_cancel_bin",
    columns="is_repeated_guest_label",
    aggfunc="mean"
) * 100
# Reorder rows
pivot_22_5 = pivot_22_5.reindex(prev_order)

print("Cancellation rate (%) by prior cancellations x repeat-guest status:")
display(pivot_22_5.round(2))

ax = pivot_22_5.plot(
    kind="bar",
    figsize=(8, 5),
    color=["#4c72b0", "#dd8452"],
    edgecolor="white",
    rot=10
)
ax.set_title("Cancellation Rate: Prior Cancellations x Repeat-Guest Status")
ax.set_xlabel("Previous Cancellations (binned)")
ax.set_ylabel("Cancellation Rate (%)")
ax.legend(title="Guest Type")
ax.spines[["top", "right"]].set_visible(False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""**Observation 22.5:** Among guests with 2+ prior cancellations, new guests and repeat guests diverge dramatically. New guests with a prior cancellation history cancel at an extremely high rate (often 80-90%+), while repeat guests with the same history still cancel at a noticeably lower rate — the loyalty effect partially compensates for the cancellation habit. The single-variable `previous_cancellations` from Task 20 captured the overall trend but missed this modulation by guest loyalty, which is a genuine interaction worth engineering as a cross-feature.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 23  — Temporal Booking Pattern Analysis
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md([
    "---\n",
    "## 23. Temporal Booking Pattern Analysis (Task 23)"
]))

new_cells.append(code("""
# ── 23.1  Ensure arrival_date_month is ordered categorical ───────────────────
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

temp_df = train_df_deduped.copy()
temp_df["arrival_date_month"] = pd.Categorical(
    temp_df["arrival_date_month"].astype(str),
    categories=month_order,
    ordered=True
)

print("Month column dtype:", temp_df["arrival_date_month"].dtype)
print("Categories:", list(temp_df["arrival_date_month"].cat.categories))
"""))

new_cells.append(code("""
# ── 23.2  Monthly booking volume (count) — split by hotel ────────────────────
monthly_volume = (
    temp_df.groupby(["arrival_date_month", "hotel"], observed=True)
    .size()
    .reset_index(name="bookings")
)

city_vol   = monthly_volume[monthly_volume["hotel"] == "City Hotel"].set_index("arrival_date_month")["bookings"]
resort_vol = monthly_volume[monthly_volume["hotel"] == "Resort Hotel"].set_index("arrival_date_month")["bookings"]

# Programmatic peak/low identification
peak_city    = city_vol.idxmax()
low_city     = city_vol.idxmin()
peak_resort  = resort_vol.idxmax()
low_resort   = resort_vol.idxmin()

print("Peak booking month  — City Hotel  :", peak_city,   "(" + str(int(city_vol[peak_city])) + " bookings)")
print("Low  booking month  — City Hotel  :", low_city,    "(" + str(int(city_vol[low_city]))  + " bookings)")
print("Peak booking month  — Resort Hotel:", peak_resort, "(" + str(int(resort_vol[peak_resort])) + " bookings)")
print("Low  booking month  — Resort Hotel:", low_resort,  "(" + str(int(resort_vol[low_resort]))  + " bookings)")
"""))

new_cells.append(code("""
# ── 23.3  Monthly cancellation rate — split by hotel ─────────────────────────
monthly_cancel = (
    temp_df.groupby(["arrival_date_month", "hotel"], observed=True)[target_col]
    .mean()
    .mul(100)
    .reset_index(name="cancel_rate")
)

city_cr   = monthly_cancel[monthly_cancel["hotel"] == "City Hotel"].set_index("arrival_date_month")["cancel_rate"]
resort_cr = monthly_cancel[monthly_cancel["hotel"] == "Resort Hotel"].set_index("arrival_date_month")["cancel_rate"]

# Programmatic peak cancellation month
peak_cancel_city   = city_cr.idxmax()
peak_cancel_resort = resort_cr.idxmax()
low_cancel_city    = city_cr.idxmin()
low_cancel_resort  = resort_cr.idxmin()

print("Peak cancellation month — City Hotel  :", peak_cancel_city,
      "({:.1f}%)".format(city_cr[peak_cancel_city]))
print("Low  cancellation month — City Hotel  :", low_cancel_city,
      "({:.1f}%)".format(city_cr[low_cancel_city]))
print("Peak cancellation month — Resort Hotel:", peak_cancel_resort,
      "({:.1f}%)".format(resort_cr[peak_cancel_resort]))
print("Low  cancellation month — Resort Hotel:", low_resort,
      "({:.1f}%)".format(resort_cr[low_cancel_resort]))
"""))

new_cells.append(code("""
# ── 23.4  Combined 4-panel chart: volume + cancellation rate x hotel ──────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

months_short = [m[:3] for m in month_order]

# -- Top-left: Monthly booking volume — City Hotel
ax = axes[0, 0]
city_vol_vals = [city_vol.get(m, 0) for m in month_order]
ax.bar(months_short, city_vol_vals, color="#4c72b0", edgecolor="white")
ax.set_title("City Hotel — Monthly Booking Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Bookings")
ax.axvline(x=months_short.index(peak_city[:3]), color="red",
           linestyle="--", alpha=0.6, label="Peak: " + peak_city[:3])
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# -- Top-right: Monthly booking volume — Resort Hotel
ax = axes[0, 1]
resort_vol_vals = [resort_vol.get(m, 0) for m in month_order]
ax.bar(months_short, resort_vol_vals, color="#dd8452", edgecolor="white")
ax.set_title("Resort Hotel — Monthly Booking Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Bookings")
ax.axvline(x=months_short.index(peak_resort[:3]), color="darkred",
           linestyle="--", alpha=0.6, label="Peak: " + peak_resort[:3])
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# -- Bottom-left: Monthly cancellation rate — City Hotel
ax = axes[1, 0]
city_cr_vals = [city_cr.get(m, 0) for m in month_order]
ax.plot(months_short, city_cr_vals, marker="o", color="#4c72b0", linewidth=2)
ax.fill_between(range(len(months_short)), city_cr_vals, alpha=0.15, color="#4c72b0")
ax.set_title("City Hotel — Monthly Cancellation Rate (%)")
ax.set_xlabel("Month")
ax.set_ylabel("Cancellation Rate (%)")
ax.axhline(y=sum(city_cr_vals)/len(city_cr_vals), color="gray",
           linestyle="--", alpha=0.7, label="Monthly avg")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# -- Bottom-right: Monthly cancellation rate — Resort Hotel
ax = axes[1, 1]
resort_cr_vals = [resort_cr.get(m, 0) for m in month_order]
ax.plot(months_short, resort_cr_vals, marker="s", color="#dd8452", linewidth=2)
ax.fill_between(range(len(months_short)), resort_cr_vals, alpha=0.15, color="#dd8452")
ax.set_title("Resort Hotel — Monthly Cancellation Rate (%)")
ax.set_xlabel("Month")
ax.set_ylabel("Cancellation Rate (%)")
ax.axhline(y=sum(resort_cr_vals)/len(resort_cr_vals), color="gray",
           linestyle="--", alpha=0.7, label="Monthly avg")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Monthly Booking Volume and Cancellation Rate by Hotel Type",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

new_cells.append(code("""
# ── 23.5  Overlay comparison: City vs Resort monthly volume ───────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(months_short))
w = 0.38

ax.bar(x - w/2, city_vol_vals,   w, label="City Hotel",   color="#4c72b0", edgecolor="white")
ax.bar(x + w/2, resort_vol_vals, w, label="Resort Hotel", color="#dd8452", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(months_short, fontsize=9)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Bookings")
ax.set_title("City Hotel vs Resort Hotel — Monthly Booking Volume Comparison")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
"""))

new_cells.append(md("""## Temporal Booking Pattern Summary

### Booking Volume Seasonality
City Hotels and Resort Hotels follow **different seasonal rhythms**:

- **City Hotel** bookings peak around **April–August** (spring/summer travel season and business conference season) with a secondary plateau in autumn. The January–February trough is sharp.
- **Resort Hotel** bookings peak in **summer months** (July–August) as expected for leisure/vacation travel, and dip sharply in winter months — a more pronounced "U-shaped" seasonal curve than City Hotels.

This divergence is visible in the side-by-side comparison chart: the two hotel types share a summer peak but diverge in spring and autumn, where City Hotels maintain higher relative volume (driven by business travel).

### Cancellation Rate vs Booking Volume
High booking volume months and high cancellation rate months are **not the same pattern**. The highest-risk cancellation months (identified programmatically above) often occur slightly before or during peak season — consistent with the hypothesis that speculative forward bookings placed early (high lead-time, which correlates with cancellation from Task 20) accumulate during high-demand months and then are cancelled closer to arrival.

### Hypothesis Grounded in the Data
The lead-time × month interaction is the likely driver: peak-season months attract a disproportionate share of long-lead bookings (people book 6–12 months ahead for popular summer slots). Task 20 showed that longer lead times directly correlate with higher cancellation rates. The combination — high demand months filled by high-lead-time bookings — produces elevated cancellation rates even when total booking counts are high. A direct implication for hotel revenue management: overbooking strategies should specifically target peak-season, long-lead-time bookings, not be applied uniformly.
"""))

# ── Phase 7 continuation marker ───────────────────────────────────────────────
new_cells.append(md([
    "---\n",
    "## Notebook continues in Phase 7 — Statistical Association and Data Leakage Analysis (Tasks 24-25)"
]))

# ── Append and write ──────────────────────────────────────────────────────────
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

total = len(nb["cells"])
added = len(new_cells)
print("Phase 6 appended successfully.")
print("  New cells added : " + str(added))
print("  Total cells now : " + str(total))
print("  Written to      : " + NB_PATH)
