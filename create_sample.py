"""
Make a small 500-row sample of the raw data so the repo can be tried
without downloading the full Kaggle file.

Run from the repo root:
  python create_sample.py
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/DSI_kickstarterscrape_dataset.csv")
SAMPLE_PATH = Path("sample_data/kickstarter_sample.csv")

df = pd.read_csv(RAW_PATH, encoding="latin-1")
sample = df.sample(n=500, random_state=42)
SAMPLE_PATH.parent.mkdir(exist_ok=True)
sample.to_csv(SAMPLE_PATH, index=False)
print(f"Saved {len(sample)} rows to {SAMPLE_PATH}")
