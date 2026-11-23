"""
Milestone 1 – Build the modeling dataset from the raw Kaggle CSV.

Steps (each one is logged with a row count):
  1. Keep only campaigns with a final outcome: 'successful' or 'failed'.
  1b. Fix HTML codes in labels ('Film &amp; Video' -> 'Film & Video').
  2. Keep only U.S. campaigns (location ends in a valid 2-letter state code).
  3. Split 'location' into 'city' and 'state'.
  4. Drop rows that still have missing values in the columns we keep.
  5. Add the binary target 'success' (1 = successful, 0 = failed).

Output: data/processed/kickstarter_us_clean.csv

Run from the repo root:
  python scripts/filter_campaigns.py
"""
import html
from pathlib import Path

import pandas as pd

from load_and_inspect_data import load_raw

OUT_PATH = Path("data/processed/kickstarter_us_clean.csv")

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
}


def build_clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = []
    log.append(("Raw data", len(df)))

    # 1. Final outcomes only
    df = df[df["status"].isin(["successful", "failed"])].copy()
    log.append(("Keep successful/failed", len(df)))

    # Fix HTML codes in text labels (e.g. 'Film &amp; Video' -> 'Film & Video')
    for col in ["category", "subcategory", "location"]:
        df[col] = df[col].map(lambda s: html.unescape(s) if isinstance(s, str) else s)

    # 2-3. Split location -> city, state and keep U.S. rows
    parts = df["location"].str.rsplit(",", n=1, expand=True)
    df["city"] = parts[0].str.strip()
    df["state"] = parts[1].str.strip()
    df = df[df["state"].isin(US_STATES)].copy()
    log.append(("Keep U.S. campaigns", len(df)))

    # 4. Drop remaining missing values
    df = df.dropna()
    log.append(("Drop missing values", len(df)))

    # 5. Binary target
    df["success"] = (df["status"] == "successful").astype(int)

    if verbose:
        print("Transformation log")
        for step, n in log:
            print(f"  {step:<25} {n:>7,} rows")
    return df.reset_index(drop=True)


def main() -> None:
    clean = build_clean_dataset(load_raw())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(clean):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
