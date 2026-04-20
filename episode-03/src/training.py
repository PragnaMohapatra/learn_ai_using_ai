"""Training logic for Episode 3."""

import json
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


REQUIRED_COLUMNS = {"text", "label"}
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def _read_input_dataset(input_path: str) -> pd.DataFrame:
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {input_path}")

    if dataset_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(dataset_path)
    else:
        dataframe = pd.read_csv(
            dataset_path,
            sep="\t",
            header=None,
            names=["label", "text"],
        )

    if dataframe.empty:
        raise ValueError("Training dataset is empty.")

    return dataframe


def clean_training_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training dataset is missing required columns: {missing_list}")

    cleaned = dataframe.loc[:, ["text", "label"]].copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    cleaned["label"] = cleaned["label"].astype("string").str.strip().str.lower()
    cleaned = cleaned[(cleaned["text"] != "") & cleaned["label"].notna() & (cleaned["label"] != "")]
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Training dataset has no usable rows after cleaning.")

    return cleaned


def prepare_training_data(input_path: str, output_csv: str) -> dict:
    source = _read_input_dataset(input_path)
    cleaned = clean_training_dataframe(source)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    return {
        "status": "data_prepared",
        "input_path": input_path,
        "output_csv": str(output_path),
        "rows": int(len(cleaned)),
        "columns": list(cleaned.columns),
        "labels": sorted(cleaned["label"].astype(str).unique().tolist()),
    }


def load_training_data(input_csv: str) -> pd.DataFrame:
    return clean_training_dataframe(_read_input_dataset(input_csv))


def split_training_data(
    dataframe: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    if len(dataframe) < 2:
        raise ValueError("Need at least 2 rows to split training data.")

    X = dataframe["text"]
    y = dataframe["label"]

    stratify = None
    class_count = int(y.nunique())
    test_rows = int(test_size) if isinstance(test_size, int) else ceil(len(dataframe) * test_size)
    train_rows = len(dataframe) - test_rows
    if (
        class_count > 1
        and y.value_counts().min() >= 2
        and test_rows >= class_count
        and train_rows >= class_count
    ):
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "stratified": stratify is not None,
    }


def vectorize_text(
    X_train: pd.Series,
    X_test: pd.Series,
    *,
    max_features: int = 2000,
) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    return {
        "vectorizer": vectorizer,
        "X_train_features": X_train_features,
        "X_test_features": X_test_features,
        "train_shape": tuple(X_train_features.shape),
        "test_shape": tuple(X_test_features.shape),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }


def train_baseline_models(
    X_train_features,
    X_test_features,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    if y_train.nunique() < 2:
        raise ValueError("Need at least 2 label classes in the training split to train the classifiers.")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
        "multinomial_nb": MultinomialNB(),
    }

    model_results = {}
    fitted_models = {}
    best_model = None
    best_accuracy = -1.0

    for name, model in candidates.items():
        model.fit(X_train_features, y_train)
        predictions = model.predict(X_test_features)
        accuracy = float(accuracy_score(y_test, predictions))
        model_results[name] = {
            "accuracy": round(accuracy, 4),
            "predictions": predictions.tolist(),
        }
        fitted_models[name] = model
        if accuracy > best_accuracy:
            best_model = name
            best_accuracy = accuracy

    return {
        "models": model_results,
        "fitted_models": fitted_models,
        "best_model": best_model,
        "best_accuracy": round(best_accuracy, 4),
    }


def train_classifier(input_csv: str, output_dir: str) -> dict:
    """
    First training step: load and validate labeled text data.

    Later this will:
    - split train/test
    - vectorize text
    - train a classifier
    - save the model
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = load_training_data(input_csv)
    labels = sorted(dataframe["label"].astype(str).unique().tolist())
    split_result = split_training_data(dataframe)
    vectorized = vectorize_text(split_result["X_train"], split_result["X_test"])
    model_results = train_baseline_models(
        vectorized["X_train_features"],
        vectorized["X_test_features"],
        split_result["y_train"],
        split_result["y_test"],
    )

    best_model_name = model_results["best_model"]
    best_model_obj = model_results["fitted_models"][best_model_name]

    model_file = output_path / "best_model.joblib"
    vectorizer_file = output_path / "vectorizer.joblib"
    metrics_file = output_path / "metrics.json"

    joblib.dump(best_model_obj, model_file)
    joblib.dump(vectorized["vectorizer"], vectorizer_file)

    metrics = {
        "labels": labels,
        "train_rows": split_result["train_rows"],
        "test_rows": split_result["test_rows"],
        "test_size": split_result["test_size"],
        "random_state": split_result["random_state"],
        "stratified": split_result["stratified"],
        "vocabulary_size": vectorized["vocabulary_size"],
        "model_scores": {
            name: {"accuracy": details["accuracy"]}
            for name, details in model_results["models"].items()
        },
        "best_model": best_model_name,
        "best_accuracy": model_results["best_accuracy"],
    }
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "status": "model_saved",
        "input_csv": input_csv,
        "output_dir": str(output_path),
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "text_column": "text",
        "label_column": "label",
        "labels": labels,
        "train_rows": split_result["train_rows"],
        "test_rows": split_result["test_rows"],
        "test_size": split_result["test_size"],
        "random_state": split_result["random_state"],
        "stratified": split_result["stratified"],
        "train_shape": vectorized["train_shape"],
        "test_shape": vectorized["test_shape"],
        "vocabulary_size": vectorized["vocabulary_size"],
        "model_scores": model_results["models"],
        "best_model": best_model_name,
        "best_accuracy": model_results["best_accuracy"],
        "model_file": str(model_file),
        "vectorizer_file": str(vectorizer_file),
        "metrics_file": str(metrics_file),
    }