# Kickstarter Crowdfunding Recommendation Engine

Built for a **State Street** challenge in the **Break Through Tech AI Studio (Fall 2026)** program. The project predicts whether a Kickstarter campaign will succeed or fail using information available at launch, explains which factors drive those predictions, and will prototype recommendations for campaigns at risk of failing.

📋 **Project board:** [paste your GitHub Project link here]

---

## 👩🏻‍💻 Author

| Name | GitHub Handle | Contribution |
|------|---------------|--------------|
| Michelle Lin | @[your-handle] | Individual project: all milestones (data preparation, EDA, modeling, recommendation prototype) |

> This project started as a team project and became an individual project.

---

## 🎯 Project Highlights

- Built a reproducible pipeline that turns **45,957 raw campaigns** into **38,463 U.S. campaigns** with a final success/fail outcome.
- Cleaned inconsistent category labels (e.g. `Film &amp; Video` vs. `Film & Video`) and removed **data-leakage** columns (pledged, backers, comments, funded date).
- Engineered features from reward tiers (min and max reward) and encoded categorical variables, giving a **70-column model-ready dataset**.
- EDA found that **goal size, campaign length, category, and reward tiers** all relate strongly to success. **Updates** show the strongest link but are flagged as possible leakage.

---

## 👩🏽‍💻 Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/kickstarter-success-predictor.git
cd kickstarter-success-predictor
```

**2. Create a virtual environment** (Python 3.11+)
```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
# Windows PowerShell
python -m venv .venv; .venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Get the data**
Download the Kickstarter Projects dataset from Kaggle and save it as:
```
data/DSI_kickstarterscrape_dataset.csv
```
*(Optional:* `sample_data/kickstarter_sample.csv` *has 500 rows to try things quickly.)*

**5. Build the clean dataset**
```bash
python scripts/load_and_inspect_data.py   # optional: inspect the raw file
python scripts/filter_campaigns.py        # writes data/processed/kickstarter_us_clean.csv
```

**6. Run the notebooks in order** (Jupyter or VS Code)

| Notebook | Milestone | What it does |
|----------|-----------|--------------|
| `notebooks/milestone1_data_preparation.ipynb` | 1 – Data Preparation | Filtering, U.S. records, missing values, label cleanup, leakage removal, encoding |
| `notebooks/milestone2_eda.ipynb` | 2 – EDA | Success vs. goal, duration, updates, category, state, reward levels; correlations |
| `notebooks/milestone3_*.ipynb` | 3 – Machine Learning | *Coming next* |
| `notebooks/milestone4_*.ipynb` | 4 – Recommendation Prototype | *Planned* |

---

## 🏗️ Project Overview

Crowdfunding creators often set goals and campaign lengths without knowing what tends to work. This project uses past Kickstarter data to (1) predict whether a campaign will succeed and (2) turn the patterns behind successful campaigns into practical advice, such as setting a more realistic goal or a shorter campaign.

**ML problem type:** supervised binary classification (1 = successful, 0 = failed).
**Success criteria:** about 80%+ accuracy, plus strong precision, recall, F1, and ROC-AUC.
**Benchmark to reproduce:** Random Forest, about 80.2% accuracy, F1 = 0.834 (original project).

---

## 📊 Data Exploration

**Dataset:** Kickstarter Projects (Kaggle), 45,957 rows × 17 columns (CSV).

### Data preparation log

| Step | Rows after |
|------|-----------|
| Raw data | 45,957 |
| Keep `successful` / `failed` (drop live, canceled, suspended) | 41,965 |
| Fix HTML codes in labels (14 → 13 categories) | 41,965 |
| Keep U.S. campaigns (valid 2-letter state), split location into city and state | 38,504 |
| Drop rows with missing values | **38,463** |

**Leakage removed:** `pledged`, `funded percentage`, `backers`, `comments`, and `funded date` are only known after a campaign ends.
**Encoding:** one-hot for `category` and `state`, frequency encoding for `subcategory`; `city` dropped (3,140 values).
**New features:** `min_reward` and `max_reward` parsed from the reward-levels text.

### Key EDA findings
Overall success rate: **54.8%** (classes are fairly balanced).

| Factor | Finding |
|--------|---------|
| Goal | Success drops from about 69% (<$1k) to about 13% ($50k+) |
| Duration | Under 30 days: about 65%. Over 60 days: about 42% |
| Updates | 0 updates: about 20%. 10+: about 91% (may be leakage) |
| Category | Dance, Theater, Music highest (68–76%); Fashion, Technology, Publishing lowest (32–39%) |
| Reward tiers | 1–3 tiers: about 36%. 13+ tiers: about 69% |
| State | CA and NY make up about 38% of campaigns; state matters less than goal or category |

Charts are saved in `figures/`.

### Challenges
- **Data leakage:** several columns are only known after the campaign ends.
- **`updates` timing:** creators post updates during the campaign, so the feature may not be fully available at launch.
- **Skewed goals:** some goals reach $21M, so ranges and rank-based correlation were used.

---

## 🧠 Model Development *(Milestone 3, in progress)*

Plan:
- Stratified 80/20 train/test split
- Baseline: Logistic Regression; compare with KNN, Random Forest, and XGBoost
- Train each model **with and without `updates`** to measure leakage risk
- Tune with `GridSearchCV`, explain with feature importance and SHAP

---

## 📈 Results & Key Findings

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | | | | | |
| KNN | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

---

## 🚀 Next Steps

- Finish Milestone 3 modeling and fill in the results table
- Build the Milestone 4 recommendation prototype for at-risk campaigns
- Limitations to address: association vs. causation, the timing of `updates`, and the effect of using a single test split

---

## 📝 License

To be confirmed with the Challenge Advisor.

---

## 📄 References

- [Kickstarter](https://www.kickstarter.com/)
- Kickstarter Projects dataset (Kaggle)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [SHAP documentation](https://shap.readthedocs.io/)

---

## 🙏 Acknowledgements

Thank you to Challenge Advisor **Parth Rana** and AI Studio Coach **Darshan Ugale** for their guidance, and to Break Through Tech AI Studio.
