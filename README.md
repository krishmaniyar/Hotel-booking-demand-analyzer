# Hotel Booking Demand — Exploratory Data Analysis

Exploratory data analysis on the Hotel Booking Demand dataset, done as part of a Data Science course assignment. The goal was to understand the dataset well enough to make informed decisions before any model gets built — not to build a model yet.

## Key numbers, for reference

| Metric | Value |
|---|---|
| Rows / columns | 119,390 / 32 |
| Training split | 95,512 rows (80%, stratified) |
| Target balance | ~63% not cancelled / ~37% cancelled (1.6:1) |
| Missing data concentrated in | `company` (~94%), `agent` (~14%) |
| Strongest categorical predictor | `deposit_type` (Cramér's V = 0.4827) |
| Strongest numerical predictor | `previous_cancellations` |
| Leakage AUC gap | 0.7299 (clean baseline) vs 1.0000 (with `reservation_status`) |
| Features engineered | 8 |
| Features dropped for leakage | `reservation_status`, `reservation_status_date` |

## Table of Contents
1. [Dataset](#dataset)
2. [Running it](#running-it)
3. [What's actually in the notebook](#whats-actually-in-the-notebook)
   - [Getting oriented](#getting-oriented)
   - [Data quality](#data-quality)
   - [Missing values](#missing-values)
   - [Univariate analysis](#univariate-analysis)
   - [Target variable](#target-variable)
   - [Outliers](#outliers)
   - [Bivariate and multivariate analysis](#bivariate-and-multivariate-analysis)
   - [Temporal patterns](#temporal-patterns)
   - [Statistical testing](#statistical-testing)
   - [Data leakage](#data-leakage)
   - [Feature engineering](#feature-engineering)
   - [Preprocessing recommendations](#preprocessing-recommendations)
4. [What I'd do differently with more time](#what-id-do-differently-with-more-time)
5. [Structure](#structure)

## Dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), originally published by Antonio, Almeida and Nunes in *Data in Brief* (2019). 119,390 reservation records from a City Hotel and a Resort Hotel, covering arrivals between 2015 and 2017, 32 columns. The prediction target is `is_canceled` — whether the booking was eventually cancelled.

The CSV is included in `data/` so the notebook runs without any manual download step. If you'd rather pull it fresh, it's also available directly through Kaggle's "Add Input" on a Kaggle notebook, or via `kagglehub.dataset_download("jessemostipak/hotel-booking-demand")`.

## Running it

The notebook auto-detects its environment — it checks for the Kaggle input path first, then falls back to `data/hotel_bookings.csv` locally, so it runs unmodified in both places.

- **Kaggle**: upload `ML_Ex01_EDA.ipynb`, add the `hotel-booking-demand` dataset as input, Run All.
- **Local**: `pip install pandas numpy matplotlib seaborn scipy scikit-learn`, then run the notebook normally. The dataset is already in `data/`.

## What's actually in the notebook

I split the analysis into 29 tasks, roughly following the standard EDA sequence: get to know the raw data, check its quality, look at each feature on its own, then look at how features relate to each other and to the target, and finally turn all of that into concrete preprocessing decisions.

### Getting oriented
Before touching anything I split off 20% of the data as a stratified holdout and did the rest of the EDA only on the training portion — the assignment brief was explicit that decisions about missing values, outliers, and encoding shouldn't be influenced by data the eventual model won't get to see either. I also built a full data dictionary early, since a few columns (`agent`, `company`, `reservation_status`) needed flags before I even got to the analysis proper.

### Data quality
A small number of exact duplicate rows show up (well under 0.2%). I kept them — this dataset logs individual booking events, not unique guests, so two different guests legitimately booking the same room type, dates, and rate isn't an error, it's just two rows that happen to be identical. I checked whether the cancellation rate differed between duplicated and non-duplicated rows before deciding this, rather than assuming.

`children`, `agent`, and `company` load as `float64` purely because missing values force pandas out of integer dtype — I converted them to nullable `Int64` to keep them behaving like the IDs and counts they actually are. A handful of impossible rows exist too: negative ADR (physically meaningless, has to be a data entry error) and a few zero-occupancy bookings (zero adults, zero children, zero babies), which I flagged rather than silently dropped.

### Missing values
Two columns account for almost all of the missing data: `company` (~94% missing) and `agent` (~14% missing). Neither is random. Cross-tabulating missingness against `distribution_channel` shows agent and company are essentially always missing for direct bookings, where by definition no agency or company is involved — that's Missing At Random, not Missing Completely At Random, and it changes how I'd impute it (a "no agent" indicator is more honest than mean-filling a made-up agent ID). `country` and `children` are missing in tiny, seemingly unrelated numbers — closer to MCAR, safe to impute simply.

![Missing Value Analysis](readme_assets/08_missing_value_identification_a.png)

### Univariate analysis
Most of the numerical features are skewed rather than normal — `lead_time`, `previous_cancellations`, and `days_in_waiting_list` in particular are heavily right-skewed with a long tail of extreme values, which ruled out plain z-score-based outlier detection and pushed me toward IQR-based bounds and rank-based tests (Spearman, Mann-Whitney) instead of Pearson/t-tests later on. On the categorical side, `country`, `agent`, and `company` are high-cardinality (roughly 160, 320, and 340 unique values respectively) — one-hot encoding all three would add 800+ mostly-empty binary columns, so I ruled that out in favor of frequency or target encoding.

### Target variable
`is_canceled` splits roughly 63/37 (not cancelled / cancelled) — a ratio of about 1.6:1. That's moderate imbalance, not severe. I didn't reach for SMOTE or aggressive resampling because at this ratio `class_weight='balanced'` does the same correction without synthesizing fake rows, and accuracy on its own would be a misleading metric here (a model that always predicts "not cancelled" scores ~63% while learning nothing).

![Target Variable Distribution](readme_assets/16_target_distribution.png)

### Outliers
I ran IQR-based detection on the key numerical features but didn't drop anything at this stage — the point of this pass was to classify *why* the outliers exist, not remove them blindly. Lead times of 400+ days look extreme but are consistent with real group or wedding bookings made far in advance. Negative ADR, on the other hand, is not a valid outlier, it's a data error, and gets clipped in preprocessing rather than treated as a legitimate extreme value.

![Outlier Detection](readme_assets/18_outlier_visualization.png)

### Bivariate and multivariate analysis
No pair of numerical features showed problematic multicollinearity (nothing crossed |r| > 0.7). On the categorical side, `deposit_type` turned out to be the strongest single predictor by Cramér's V — and in a counterintuitive direction: bookings with a "Non-Refund" deposit cancel at close to 100%, the opposite of what you'd expect if a deposit was meant to discourage cancellation. Digging into that with a three-way breakdown (deposit type × market segment × cancellation) confirmed it holds across nearly every market segment, not just one — this reads like a booking-pattern artifact rather than genuine guest commitment (agencies may be using non-refundable bookings for provisional or speculative holds). 

![Correlation Heatmap](readme_assets/19_numerical_numerical_relationships_a.png)

![Multivariate Interaction](readme_assets/22_multi_feature_relationship_analysis_c.png)

### Temporal patterns
City Hotel and Resort Hotel don't share a seasonal shape. City Hotel bookings peak April through August with a smaller autumn bump; Resort Hotel is a sharper summer peak with a deep winter trough. Cancellation rate doesn't track booking volume exactly either — the busiest months aren't always the highest-cancellation months, which suggests lead time (long-lead bookings accumulate disproportionately in peak season and cancel more) is doing some of that work rather than seasonality on its own.

![Temporal Analysis](readme_assets/23_temporal_booking_pattern_analysis_a.png)

### Statistical testing
With roughly 95,000 rows in the training set, almost every feature comes back statistically significant (p < 0.05) whether or not it's actually useful — large-N significance is basically guaranteed. So I leaned on effect size (point-biserial |r| for numerical features, Cramér's V for categorical) rather than p-values to decide what mattered, and said so explicitly in the notebook rather than reporting p-values as if they settled anything.

### Data leakage
This was the most important finding in the whole notebook. `reservation_status` almost perfectly encodes the target — "Canceled" maps to `is_canceled = 1` in 100% of rows. I didn't just assert this, I demonstrated it: a logistic regression trained with `reservation_status` included hits a ROC-AUC of 1.0000, while the same model without it lands around 0.7299. That ~0.27 AUC gap is not a better model, it's the model reading the answer off a feature that already contains it. `reservation_status_date` has the same problem for a different reason — it's the date the final status was recorded, which for a cancellation is at or after the cancellation itself, so it can't be known at prediction time. Both features get dropped before any real model gets trained.

### Feature engineering
I built eight new features off the back of what the EDA surfaced — things like `room_match` (did the guest get the room type they actually booked), `prev_cancel_ratio` (a smoothed version of prior cancellation history), and `is_long_lead_time` (using the actual IQR outlier threshold computed earlier, not an arbitrary cutoff). I checked each one's correlation with the target rather than assuming it would help — `prev_cancel_ratio` came out with a stronger point-biserial correlation than any single raw feature, which is a decent sign the engineering was worth doing rather than just adding noise.

### Preprocessing recommendations
The last piece of the notebook ties all of the above into one table: what to impute and how, what to clip or transform, what to encode and with which method, and what to drop outright. It's meant to be read on its own by whoever picks this up next.

## What I'd do differently with more time

The country/agent/company encoding decision (frequency vs target encoding) is left as a recommendation rather than implemented and compared — that's a reasonable next step once this moves into actual modeling. I also didn't get into interaction terms beyond the five three-way breakdowns in the multivariate section; there's likely more structure in combinations like `market_segment × customer_type` that a tree-based model would pick up automatically but that a linear model would need engineered explicitly.

## Structure

```
.
├── ML_Ex01_EDA.ipynb     # the full analysis
├── data/
│   └── hotel_bookings.csv
├── readme_assets/        # extracted visualizations
└── README.md
```