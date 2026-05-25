from typing import Dict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config.loader import load_config

def make_preprocessor() -> ColumnTransformer:
    """
    Creates and returns a ColumnTransformer that performs all required
    preprocessing steps for the phishing detection dataset.

    The preprocessing pipeline includes:
      - Numeric features: median imputation followed by robust scaling to
        mitigate the effect of extreme outliers and heavy-tailed distributions.
      - Categorical features: imputation using the most frequent category
        followed by one-hot encoding with graceful handling of unseen categories.
      - Binary or passthrough features: included without transformation.

    This function constructs the preprocessing logic but does not fit it.
    The returned transformer is intended to be used inside a scikit-learn
    Pipeline so that preprocessing is fitted only on the training split,
    preventing any data leakage during cross-validation or evaluation.

    Returns:
        ColumnTransformer:
            A scikit-learn ColumnTransformer that encapsulates all
            preprocessing steps for numeric, categorical, and passthrough
            feature groups.
    """

    # determine feature groups
    categorical_features = ['Industry', 'HostingProvider']
    robust_scale_features = ['LineOfCode', 'LargestLineLength', 'NoOfImage', 'NoOfSelfRef', 'NoOfExternalRef', 'DomainAgeMonths']
    binary_features_passthrough = ['NoOfURLRedirect', 'NoOfSelfRedirect', 'NoOfPopup', 'NoOfiFrame', 'Robots', 'IsResponsive']

    # create transformers and column transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, robust_scale_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('passthrough', 'passthrough', binary_features_passthrough)
        ],
        remainder='drop' # Drop any columns not specified
    )

    return preprocessor

def get_models() -> Dict[str, Pipeline]:
    """
    Constructs and returns a collection of machine learning models wrapped in
    scikit-learn Pipelines, each paired with the dataset's preprocessing
    transformer.

    This function defines a set of baseline classification models that are
    suitable for the phishing detection task, including both linear and
    tree-based approaches. Each model is combined with a fresh preprocessing
    pipeline created via `make_preprocessor()`, ensuring that preprocessing
    steps (imputation, encoding, scaling) are fitted only on the training data
    during model training or cross-validation. This design prevents data
    leakage and keeps each model fully self-contained.

    Returns:
        dict[str, Pipeline]:
            A dictionary mapping model names to sklearn Pipeline objects,
            where each Pipeline consists of:
              - "preprocessor": the ColumnTransformer returned by
                `make_preprocessor()`
              - "clf": an initialized sklearn classifier

            Example keys: "Logistic Regression", "Random Forest",
            "Gradient Boosting".
    """
    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("clf", LogisticRegression())
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("clf", RandomForestClassifier())
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", make_preprocessor()),
            ("clf", GradientBoostingClassifier())
        ])
    }

def train_models(models: Dict[str, Pipeline], X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Pipeline]:
    """
    Trains a collection of machine learning models using the training dataset,
    optionally applying hyperparameter tuning based on the configuration
    specified in the external YAML file.

    This function loads the hyperparameter settings from the project's
    configuration file and, for each model, determines whether tuning should be
    applied. If hyperparameter distributions are provided for a model, the
    model's Pipeline (including preprocessing and classifier) is wrapped in a
    RandomizedSearchCV object and trained using cross-validation on the training
    data. The best estimator found through the search is retained. If no tuning
    parameters exist for a model, it is simply trained using its default
    configuration.

    By running RandomizedSearchCV on the full Pipeline, the preprocessing steps
    (imputation, encoding, scaling) are correctly refit within each CV fold,
    preventing data leakage between folds and ensuring robust model selection.

    Args:
        models (dict[str, Pipeline]):
            A dictionary mapping model names to sklearn Pipeline objects, each
            containing a preprocessing transformer and a classifier.
        X_train (array-like or DataFrame):
            The feature matrix for the training split.
        y_train (array-like or Series):
            The label vector for the training split.

    Returns:
        dict[str, Pipeline]:
            A dictionary mapping model names to their final trained estimators.
            For tuned models, this is the best estimator found by
            RandomizedSearchCV; for untuned models, this is the fitted Pipeline.
    """
    
    # load hyperparameter settings
    config = load_config()
    param_dist = config["models"]
    tuning = config["tuning"]

    # train models and hyperparameter tuning
    trained_models = {}
    for model_name, model in models.items():
        # if model has hyperparameter distributions, apply hyperparameter tuning
        if model_name in param_dist and param_dist[model_name]:
            model = RandomizedSearchCV(model, param_dist[model_name], **tuning)
            search = model.fit(X_train, y_train)
            trained_models[model_name] = search.best_estimator_
            print(f"[{model_name}] best params: {search.best_params_}")

        else:
            print(f"[{model_name}] no hyperparameter tuning. Using default parameters.")
            model.fit(X_train, y_train)
            trained_models[model_name] = model
    
    return trained_models

def evaluate_models(trained_models: Dict[str, Pipeline], X_test: np.ndarray, y_test: pd.Series) -> None:
    """
    Evaluates a collection of trained classification models on the test set and
    prints standard performance metrics for each model.

    This function iterates through a dictionary of trained models and computes
    a suite of evaluation metrics commonly used in binary classification,
    including accuracy, precision, recall, F1-score, and ROC-AUC. A confusion
    matrix is also displayed to provide additional insight into each model's
    error patterns. These metrics help compare models and identify which one
    generalizes best to unseen data.

    Args:
        trained_models (dict[str, sklearn.base.BaseEstimator]):
            A dictionary mapping model names to trained model instances
            (typically sklearn Pipelines containing preprocessing and classifier).
        X_test (array-like or sparse matrix):
            The preprocessed feature matrix for the held-out test set.
        y_test (array-like):
            True labels corresponding to the test set.

    Returns:
        None:
            The function prints evaluation results to the console for each model
            but does not return explicit values.
    """

    for model_name, model in trained_models.items():
        # make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # print evaluation results
        print(f"=== Model Evaluation: {model_name} ===")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
        print("Precision:", round(precision_score(y_test, y_pred), 4))
        print("Recall   :", round(recall_score(y_test, y_pred), 4))
        print("F1       :", round(f1_score(y_test, y_pred), 4))
        print("ROC AUC  :", round(roc_auc_score(y_test, y_pred_proba), 4))
        print()
