"""
Step 0 – Load and inspect the raw Kickstarter dataset.

How to get the data:
  1. Go to the Kickstarter Projects dataset on Kaggle.
  2. Download the CSV and save it as: data/DSI_kickstarterscrape_dataset.csv

Run from the repo root:
  python scripts/load_and_inspect_data.py
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/DSI_kickstarterscrape_dataset.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. The file is not UTF-8, so latin-1 is used."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle CSV and save it there."
        )
    return pd.read_csv(path, encoding="latin-1")


def main() -> None:
    df = load_raw()
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    print("Column types:")
    print(df.dtypes.to_string(), "\n")
    print("Campaign status counts:")
    print(df["status"].value_counts().to_string(), "\n")
    print("Missing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())


if __name__ == "__main__":
    main()
