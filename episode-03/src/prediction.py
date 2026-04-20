"""Prediction logic for Episode 3."""

from pathlib import Path

import joblib


def load_artifacts(model_dir: str) -> dict:
    artifacts_path = Path(model_dir)
    model_file = artifacts_path / "best_model.joblib"
    vectorizer_file = artifacts_path / "vectorizer.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not vectorizer_file.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_file}")

    return {
        "model": joblib.load(model_file),
        "vectorizer": joblib.load(vectorizer_file),
        "model_file": str(model_file),
        "vectorizer_file": str(vectorizer_file),
    }


def predict_text(model_dir: str, text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    artifacts = load_artifacts(model_dir)
    features = artifacts["vectorizer"].transform([text])
    prediction = artifacts["model"].predict(features)[0]

    return {
        "model_dir": model_dir,
        "text": text,
        "prediction": str(prediction),
    }
