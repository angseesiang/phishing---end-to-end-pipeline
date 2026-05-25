from data_loader import load_data
from sklearn.model_selection import train_test_split
from model import get_models, train_models, evaluate_models

def ml_pipeline() -> None:
    """
    Executes the complete end-to-end machine learning workflow for phishing
    detection, as required for the AISG technical assessment.

    This function orchestrates all major pipeline components, including:
      1. Data loading from the SQLite database.
      2. Train–test splitting with stratification to preserve class balance.
      3. Construction of model Pipelines, each containing a preprocessing
         transformer (imputation, encoding, scaling) and a classifier.
      4. Optional hyperparameter tuning for selected models using
         RandomizedSearchCV, with search spaces and settings defined in an
         external YAML configuration file.
      5. Training of final models using the best hyperparameters obtained
         from tuning (or default settings for untuned models).
      6. Evaluation of all trained models on the held-out test set using
         metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and a
         confusion matrix.

    By delegating data preprocessing and model fitting to scikit-learn Pipelines,
    this function ensures that all preprocessing steps are fit only on the
    training split, preventing data leakage and ensuring proper cross-validation
    behavior during hyperparameter search.

    The function serves as the main entry point for the ML system and is
    designed to be invoked by the project's `run.sh` script. All outputs are
    printed to the console for transparency and for ease of assessment.

    Returns:
        None:
            The function prints progress updates, hyperparameter tuning results,
            and evaluation metrics, but does not return explicit values.
    """

    print("====== Starting ML Pipeline ======")
    # 1. Load the SQLite database
    df = load_data()

    # 2. Split the data into training and testing sets
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print("=== Dataset Detail ===")
    print("Dataset Size     :", len(df))
    print("Training Set Size:", len(X_train))
    print("Test Set Size    :", len(X_test))
    print()

    # 3. Data Preprocessing and Models
    models = get_models()

    # 4. Model Training
    trained_models = train_models(models, X_train, y_train)

    # 5. Model Evaluation
    evaluate_models(trained_models, X_test, y_test)

    print("====== Pipeline Completed ======")


if __name__ == "__main__":
    # run the ML pipeline
    ml_pipeline()
