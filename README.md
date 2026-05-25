# Phishing Website Detection Pipeline

This repository contains an end-to-end machine learning pipeline for detecting phishing websites using structured website and hosting-related features stored in a SQLite database.

The project is designed to be modular, configurable, and reproducible. It covers the full workflow from data loading and preprocessing to model training, hyperparameter tuning, and evaluation.

## Project Structure

```text
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── mlp.py
│   └── config/
│       ├── loader.py
│       └── hyperparameter.yaml
├── .gitignore
├── eda.ipynb
├── requirements.txt
├── run.sh
└── README.md
```

## Project Overview

The goal of this project is to build a machine learning pipeline that classifies websites as either phishing or legitimate based on website metadata, page structure, hosting information, and other engineered features.

The pipeline is built around the following principles:

- **Modularity**: Core tasks such as loading, preprocessing, modeling, and configuration are separated into dedicated files.
- **Reproducibility**: Random seeds and controlled configuration settings are used to make experiments more consistent.
- **Leakage prevention**: Preprocessing is handled inside machine learning pipelines to reduce the risk of data leakage.
- **Configurability**: Hyperparameter tuning settings are managed through a YAML configuration file.
- **Evaluation focus**: Metrics such as recall and F1-score are emphasized because phishing detection is sensitive to false negatives.

## Pipeline Flow

```text
Load SQLite Dataset
        ↓
Clean and Validate Data
        ↓
Train-Test Split
        ↓
Build Preprocessing Pipeline
        ↓
Train Classification Models
        ↓
Optional Hyperparameter Tuning
        ↓
Evaluate Model Performance
```

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place the `phishing.db` SQLite database file inside the `data/` directory:

```text
data/phishing.db
```

### 3. Run the Pipeline

Run the project using the provided shell script:

```bash
./run.sh
```

If needed, make the script executable first:

```bash
chmod +x run.sh
```

## Configuration

Hyperparameter tuning settings are stored in:

```text
src/config/hyperparameter.yaml
```

Example configuration structure:

```yaml
tuning:
  n_iter: 5
  cv: 3
  scoring: f1

models:
  Random Forest:
    clf__n_estimators: [50, 100, 200]
    clf__max_depth: [null, 5, 10]
    clf__min_samples_split: [2, 5, 10]
```

You can adjust model parameters, cross-validation settings, scoring metrics, and tuning iterations from this file without changing the main pipeline code.

## Exploratory Data Analysis Summary

The exploratory data analysis notebook, `eda.ipynb`, examines the dataset structure, missing values, feature distributions, label balance, and potential relationships between website attributes and phishing risk.

Key observations include:

- The dataset contains structured website-level features and a binary target label.
- Most features are numeric, while selected columns such as `Industry` and `HostingProvider` are categorical.
- `LineOfCode` contains missing values and is handled with median imputation.
- `NoOfImage` may contain invalid negative values and is corrected during data loading.
- Several numeric variables are skewed, which motivates the use of robust scaling.
- The classification target is slightly imbalanced, making recall and F1-score important evaluation metrics.

## Feature Processing

| Feature Group | Example Columns | Processing Applied | Purpose |
| --- | --- | --- | --- |
| Data correction | `NoOfImage` | Clip negative values to zero | Fix invalid values |
| Numeric features | `LineOfCode`, `LargestLineLength`, redirect counts | Median imputation and robust scaling | Handle missing values, skew, and outliers |
| Categorical features | `Industry`, `HostingProvider` | Most-frequent imputation and one-hot encoding | Convert categories into model-ready features |
| Binary/count features | `Robots`, `IsResponsive` | Passthrough where appropriate | Preserve already usable inputs |

## Models Used

The pipeline compares multiple model families to evaluate different approaches to phishing detection:

### Logistic Regression

Logistic Regression is used as a strong baseline model. It is fast, interpretable, and effective for high-dimensional feature spaces, especially after one-hot encoding categorical variables.

### Random Forest

Random Forest is included as a nonlinear tree-based model. It can capture feature interactions, handle noisy data, and provide useful feature-importance insights.

### Gradient Boosting

Gradient Boosting is included as a performance-oriented model for structured tabular data. It can capture subtle patterns by sequentially improving on earlier model errors.

## Model Evaluation

The following metrics are used to evaluate classification performance:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For phishing detection, recall is especially important because misclassifying a phishing website as legitimate can create a higher security risk than flagging a legitimate website for review. F1-score is also useful because it balances precision and recall.

## Deployment Considerations

Potential production considerations include:

- Monitoring feature drift across website and hosting patterns.
- Retraining the model periodically as phishing behavior changes.
- Serializing trained models with `joblib` for reuse.
- Ensuring preprocessing steps are bundled with the model pipeline.
- Handling unseen categorical values safely during inference.

## Notes

- The project expects the SQLite database to be available under the `data/` directory.
- Hyperparameter tuning is controlled through the YAML configuration file.
- The pipeline is designed for experimentation, learning, and reproducible model development.
