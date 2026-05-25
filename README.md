# CyberProtect: Phishing Website Detection System

## 1. My Information
- **Full Name:** Ang See Siang
- **Email:** ang_see_siang@yahoo.com  

---

## 2. Project Overview

This repository contains my submission for **AIAPⓇ Batch 22 Technical Assessment**.  
The goal is to design a **modular, configurable, and reproducible machine learning pipeline** for phishing website detection.

The pipeline is built with the following principles:

- **Modularity**: Each component (loading, preprocessing, modeling, tuning) lives in its own file.
- **Reproducibility**: Fixed random seeds, deterministic flows.
- **No data leakage**: Preprocessing happens inside Pipelines.
- **Configurability**: Hyperparameters controlled through a YAML file.
- **Explainability**: Clear evaluation metrics and meaningful modeling choices.

---

## 3. Folder Structure Overview

```
├── .github/
│   └── workflows/
│       └── github-actions.yml    # GitHub Actions CI/CD workflow
│
├── src/
│   ├── data_loader.py            # Reads SQLite database and cleans NoOfImage
│   ├── model.py                  # Data Preprocessing, Model creation, HPO, training, evaluation
│   ├── mlp.py                    # Main end-to-end ML pipeline
│   └── config/
│       ├── loader.py             # Loads YAML config safely
│       └── hyperparameter.yaml   # Config for RandomizedSearchCV tuning
│
├── .gitignore                    # Git ignore file
├── eda.ipynb                     # Task 1 Exploratory Data Analysis
├── requirements.txt              # Python dependencies
├── run.sh                        # Shell script to execute pipeline
└── README.md                     # This file
```

---

## 4. How to Run the Pipeline

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Ensure the Database Exists
Place the provided `phishing.db` file inside the `data/` folder:

```
data/phishing.db
```

### Step 3 — Run the Pipeline
Using the provided shell script:

```bash
./run.sh
```

---

## 5. Modifying Parameters (Hyperparameter Tuning)

Hyperparameters and tuning settings are configured via:

```
src/config/hyperparameter.yaml
```

Example:
```yaml
tuning:
  n_iter: 5
  cv: 3
  scoring: 'f1'

models:
  Random Forest:
    clf__n_estimators: [50, 100, 200]
    clf__max_depth: [null, 5, 10]
    clf__min_samples_split: [2, 5, 10]
    ...
```

To modify tuning behavior, simply edit this file—**no code changes required**.

---

## 6. Pipeline Flow (Logical Steps)

```
   ┌─────────────────────┐
   │ Load SQLite Dataset │
   └──────────┬──────────┘
              ▼
   ┌──────────────────────┐
   │ Train-Test Split     │
   │ (stratified 80/20)   │
   └──────────┬───────────┘
              ▼
   ┌───────────────────────────────┐
   │ Build Preprocessing Pipeline  │
   │ (impute → scale → encode)     │
   └──────────┬────────────────────┘
              ▼
   ┌───────────────────────────────┐
   │ Build ML Pipelines (LR, RF,   │
   │ Gradient Boosting)            │
   └──────────┬────────────────────┘
              ▼
   ┌───────────────────────────────┐
   │ Optional Hyperparameter Tuning│
   │ via RandomizedSearchCV/YAML   │
   └──────────┬────────────────────┘
              ▼
   ┌───────────────────────────────┐
   │ Train Final Models            │
   └──────────┬────────────────────┘
              ▼
   ┌───────────────────────────────┐
   │ Evaluate Models (metrics)     │
   └───────────────────────────────┘
```

---

## 7. Summary of Key EDA Findings (from Task 1)

### Dataset Structure
- The dataset contains 10,500 rows and 16 features
- Most features are numeric, with two categorical columns: `Industry` and `HostingProvider`

### Data Quality Observations
- `LineOfCode` is the only feature with missing values (~22%) → Median imputation chosen for robustness against skew
- `NoOfImage` contains negative values, which are invalid → Corrected by clipping values to ≥ 0 during data loading
- No other columns contain missing values

### Distribution Characteristics
- Many numeric features (e.g., `LargestLineLength`, `redirect counts`) are right-skewed → Motivated the use of `RobustScaler` to reduce outlier influence
- Categorical features have moderate cardinality, making `OneHotEncoder` appropriate.
- Several numeric variables show long-tailed distributions, common for HTML/website metrics.

### Label Imbalance
- Labels are slightly imbalanced: Legitimate (55%) and Phishing (45%)
- Chosen metrics emphasize `Recall` and `F1-score`

### Bivariate & Domain Interpretations
- Features such as `redirect counts`, `iframe usage`, and `external references` show intuitive differences between phishing and legitimate sites.
- Domain-related fields like `Industry` and `HostingProvider` display uneven distributions but remain informative.

---

## 8. Feature Processing Summary

| Feature Group              | Columns                                  | Transformation                          | Rationale |
|----------------------------|-------------------------------------------|------------------------------------------|----------|
| Data Correction            | NoOfImage                                 | Clipped to `[0, ∞)`                      | Fixes invalid negatives |
| Numeric (skewed)           | LineOfCode, LargestLineLength, etc.       | Median impute + RobustScaler             | Handle missing values and outliers |
| Categorical                | Industry, HostingProvider                 | Most-frequent impute + OneHotEncoder     | Handle missing & unseen categories |
| Binary / Count Features    | Robots, IsResponsive                      | Passthrough                              | Already clean |

---

## 9. Choice of Models

Three different classifiers were selected to provide a balanced comparison across **linear, tree-based, and boosting** approaches. The selection was guided by observations from the EDA and the characteristics of the dataset.

- **Logistic Regression**: serves as a **strong baseline model**
    - It provides a fast, interpretable linear decision boundary.
    - It handles high-dimensional sparse features well (important after `OneHotEncoding`).
    - It allows us to benchmark whether nonlinear models offer meaningful improvements.
    - It is easy to train and robust even with moderate class imbalance.
    - The dataset contains multiple numeric and categorical features, and LR provides a good check on how much linear separability exists in the phishing vs. legitimate classes.
- **Random Forest**: is chosen as a **nonlinear, tree-based model**
    - It handles feature interactions and nonlinear relationships automatically.
    - It is robust to noise, outliers, and mixed data types.
    - It provides built-in feature importance, helpful for interpretability.
    - It performs well even with minimal preprocessing.
    - EDA showed many features have skewed distributions and non-linear behavior (e.g., redirect counts, external reference counts). Random Forest can naturally capture this complexity.
    - It also benefits significantly from structured hyperparameter tuning.
- **Gradient Boosting**: is included as a **high-performance model** commonly used for structured/tabular data
    - It learns sequentially, correcting mistakes of earlier trees.
    - it often outperforms Random Forest on tabular datasets.
    - It handles complex patterns and subtle interactions effectively.
    - It is more sensitive to hyperparameters, but capable of superior results.
    - The dataset is relatively clean, medium-sized (10k rows), and contains several engineered numeric counts — an ideal scenario for boosting algorithms, which excel at exploiting small but informative signals.

---

## 10. Model Evaluation

To evaluate classification performance, the following metrics are reported:

- Confusion Matrix – Visual breakdown of true/false positives and negatives.
- Accuracy – Overall proportion of correct predictions.
- Precision – Proportion of predicted phishing sites that are truly phishing.
- **Recall** – Proportion of actual phishing sites correctly identified.
- **F1-score** – Harmonic mean of precision and recall; balances false positives and false negatives.
- ROC-AUC – Measures the model’s ability to distinguish between classes at various thresholds.

In the phishing detection domain, **Recall and F1-score** are especially critical:
- A false negative (i.e. a phishing website classified as legitimate) is much more dangerous than a false positive.
- Therefore, high recall ensures the model captures as many phishing cases as possible.
- F1-score balances this with precision, helping avoid excessive false alarms.

---

## 11. Deployment Considerations

- **Drift detection**: Monitor `Industry`/`HostingProvider` trends.
- **Retraining**: Schedule periodic model updates.
- **Serialization**: Store trained models using `joblib`.
- **Inference**: `OneHotEncoder` is robust to unseen categories (`handle_unknown="ignore"`).

---
