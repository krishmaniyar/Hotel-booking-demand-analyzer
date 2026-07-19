"""
_patch_loader.py
Replaces the dataset loader cell (index 3) in ML_Ex01_EDA.ipynb with
the full 3-step fallback chain: Kaggle path -> local path -> kagglehub auto-download.
Run: python _patch_loader.py
"""
import json, textwrap, subprocess, sys

NB_PATH     = "ML_Ex01_EDA.ipynb"
LOADER_IDX  = 3   # confirmed by inspection

# ── New loader source ──────────────────────────────────────────────────────────
NEW_LOADER = textwrap.dedent("""\
# =============================================================================
# DATASET LOADER  -- fallback chain (runs before any analysis)
# =============================================================================
# Step 1: Kaggle environment  (/kaggle/input/...)
# Step 2: Local ./data/ folder
# Step 3: kagglehub auto-download (requires Kaggle API credentials)
# If all three fail -> clean FileNotFoundError (notebook stops here, not later)
# =============================================================================

import os, shutil, subprocess, sys

KAGGLE_PATH = "/kaggle/input/hotel-booking-demand/hotel_bookings.csv"
LOCAL_PATH  = "./data/hotel_bookings.csv"
DATA_PATH   = None

# -- Step 1: Kaggle environment ------------------------------------------------
if os.path.exists(KAGGLE_PATH):
    DATA_PATH = KAGGLE_PATH
    print(f"[Step 1] Kaggle environment detected.")
    print(f"         Loading from: {DATA_PATH}")

# -- Step 2: Local ./data/ folder ----------------------------------------------
elif os.path.exists(LOCAL_PATH):
    DATA_PATH = LOCAL_PATH
    print(f"[Step 2] Local file found.")
    print(f"         Loading from: {DATA_PATH}")

# -- Step 3: kagglehub auto-download -------------------------------------------
else:
    print("[Step 1] Kaggle environment path not found.")
    print("[Step 2] Local path not found: ./data/hotel_bookings.csv")
    print("[Step 3] Attempting auto-download via kagglehub ...")

    # 3a. Ensure kagglehub is installed
    try:
        import kagglehub
        print("         kagglehub already installed.")
    except ImportError:
        print("         kagglehub not found -- installing ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kagglehub", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FileNotFoundError(
                "\\n[ERROR] Auto-download failed: could not install kagglehub.\\n"
                f"  pip stderr: {result.stderr.strip()}\\n\\n"
                "Please manually download the dataset from:\\n"
                "  https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand\\n"
                "and place hotel_bookings.csv at: ./data/hotel_bookings.csv"
            )
        import kagglehub
        print("         kagglehub installed successfully.")

    # 3b. Download dataset (needs KAGGLE_USERNAME + KAGGLE_KEY env vars
    #     or ~/.kaggle/kaggle.json with your API token)
    try:
        dl_path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
        print(f"         Download successful. Raw path: {dl_path}")

        # Find the CSV inside the returned directory (kagglehub may return a dir)
        csv_src = None
        if os.path.isdir(dl_path):
            for fname in os.listdir(dl_path):
                if fname.endswith(".csv"):
                    csv_src = os.path.join(dl_path, fname)
                    break
        elif dl_path.endswith(".csv"):
            csv_src = dl_path

        if csv_src is None:
            raise FileNotFoundError(
                f"Could not locate a .csv file inside downloaded path: {dl_path}"
            )

        # 3c. Copy to ./data/ so all subsequent cells use one consistent path
        os.makedirs("./data", exist_ok=True)
        shutil.copy2(csv_src, LOCAL_PATH)
        DATA_PATH = LOCAL_PATH
        print(f"         CSV copied to: {DATA_PATH}")

    except Exception as exc:
        raise FileNotFoundError(
            f"\\n[ERROR] Auto-download failed: {exc}\\n\\n"
            "Possible causes:\\n"
            "  - No Kaggle API credentials found.\\n"
            "    Fix: place kaggle.json in ~/.kaggle/  OR  set env vars:\\n"
            "         KAGGLE_USERNAME=<your-username>\\n"
            "         KAGGLE_KEY=<your-api-key>\\n"
            "  - No internet connection.\\n\\n"
            "Manual fix:\\n"
            "  Download from: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand\\n"
            "  Place hotel_bookings.csv at: ./data/hotel_bookings.csv"
        ) from exc

# =============================================================================
# Load dataset -- only reached if DATA_PATH is confirmed to exist above
# =============================================================================
import pandas as pd

df = pd.read_csv(DATA_PATH)
print()
print(f"Dataset loaded successfully!  Shape: {df.shape}")
""")

# ── Load notebook, replace cell, write back ───────────────────────────────────
with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

old_src = "".join(nb["cells"][LOADER_IDX]["source"])
assert "KAGGLE_PATH" in old_src, (
    f"Expected loader cell at index {LOADER_IDX} -- content mismatch. "
    "Re-run the finder script to confirm the index."
)

# Build new source list (one entry per line, '\n' on all but last)
lines = NEW_LOADER.split("\n")
nb["cells"][LOADER_IDX]["source"] = [
    l + "\n" for l in lines[:-1]
] + [lines[-1]]
nb["cells"][LOADER_IDX]["outputs"] = []
nb["cells"][LOADER_IDX]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Loader cell (index {LOADER_IDX}) patched successfully.")
print(f"New source length: {len(NEW_LOADER)} chars")
