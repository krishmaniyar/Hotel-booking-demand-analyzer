# Hotel Booking Demand — ML Exercise 01: EDA

B.Tech 4th Year Data Science assignment.  
Exploratory Data Analysis on the [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset.

## Notebook

`ML_Ex01_EDA.ipynb` — Tasks 1–3 (Dataset Loading, Data Dictionary, Train-Test Split)

## Running

### On Kaggle
1. Open a new Kaggle Notebook.
2. Add `jessemostipak/hotel-booking-demand` as an input dataset.
3. Upload `ML_Ex01_EDA.ipynb` and run all cells.

### Locally
1. Place `hotel_bookings.csv` inside the `./data/` folder.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn`
3. Open in Jupyter: `jupyter notebook ML_Ex01_EDA.ipynb`

## Structure

```
hotel-booking-demand-eda/
├── ML_Ex01_EDA.ipynb   # Main notebook (Tasks 1–3)
├── data/               # Place hotel_bookings.csv here for local runs
├── README.md
└── .gitignore
```

## Dataset

- Source: [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- 119,390 rows × 32 columns
- Target: `is_canceled` (binary classification)
