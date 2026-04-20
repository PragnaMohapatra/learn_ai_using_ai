from src import (
    prepare_training_data,
    split_training_data,
    train_baseline_models,
    train_classifier,
    vectorize_text,
)

import pandas as pd
import pytest


def test_prepare_training_data_converts_raw_sms_file_to_clean_csv(tmp_path):
    raw_path = tmp_path / "SMSSpamCollection"
    raw_path.write_text(
        "ham\t Hello there  \nspam\tWin cash now!!!\nham\tHello there\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "clean" / "sms_spam_clean.csv"

    result = prepare_training_data(str(raw_path), str(output_csv))

    assert result["status"] == "data_prepared"
    assert result["output_csv"] == str(output_csv)
    assert result["rows"] == 2
    assert result["columns"] == ["text", "label"]
    assert result["labels"] == ["ham", "spam"]

    cleaned = pd.read_csv(output_csv)
    assert list(cleaned.columns) == ["text", "label"]
    assert cleaned.loc[0, "text"] == "Hello there"
    assert cleaned.loc[0, "label"] == "ham"
    assert len(cleaned) == 2


def test_split_training_data_returns_train_and_test_sets():
    dataframe = pd.DataFrame(
        {
            "text": [
                "great service",
                "buy now",
                "love this",
                "free prize",
                "nice product",
                "urgent call",
                "highly recommend",
                "claim reward",
                "works well",
                "limited offer",
            ],
            "label": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
        }
    )

    result = split_training_data(dataframe, test_size=0.2, random_state=42)

    assert result["train_rows"] == 8
    assert result["test_rows"] == 2
    assert result["test_size"] == 0.2
    assert result["random_state"] == 42
    assert result["stratified"] is True


def test_vectorize_text_returns_feature_matrices():
    X_train = pd.Series(["great service", "love this product", "free cash offer"])
    X_test = pd.Series(["great offer", "love service"])

    result = vectorize_text(X_train, X_test, max_features=10)

    assert result["train_shape"][0] == 3
    assert result["test_shape"][0] == 2
    assert result["vocabulary_size"] > 0


def test_train_baseline_models_compares_both_models():
    X_train = pd.Series([
        "hello friend see you soon",
        "free cash prize now",
        "let us meet tomorrow",
        "claim your urgent reward",
        "happy birthday have fun",
        "win money now click here",
    ])
    y_train = pd.Series(["ham", "spam", "ham", "spam", "ham", "spam"])
    X_test = pd.Series(["free reward now", "see you later friend"])
    y_test = pd.Series(["spam", "ham"])

    vectorized = vectorize_text(X_train, X_test, max_features=50)
    result = train_baseline_models(
        vectorized["X_train_features"],
        vectorized["X_test_features"],
        y_train,
        y_test,
    )

    assert "logistic_regression" in result["models"]
    assert "multinomial_nb" in result["models"]
    assert 0.0 <= result["models"]["logistic_regression"]["accuracy"] <= 1.0
    assert 0.0 <= result["models"]["multinomial_nb"]["accuracy"] <= 1.0
    assert result["best_model"] in {"logistic_regression", "multinomial_nb"}


def test_train_classifier_creates_output_dir(tmp_path):
    input_csv = tmp_path / "reviews.csv"
    input_csv.write_text(
        "text,label\n"
        "Great product,ham\n"
        "Buy now for free prize,spam\n"
        "See you at lunch,ham\n"
        "Claim urgent reward,spam\n"
        "Thanks for your help,ham\n"
        "Win cash today,spam\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    result = train_classifier(str(input_csv), str(output_dir))

    assert result["status"] == "model_saved"
    assert result["input_csv"] == str(input_csv)
    assert result["output_dir"] == str(output_dir)
    assert result["rows"] == 6
    assert result["text_column"] == "text"
    assert result["label_column"] == "label"
    assert result["labels"] == ["ham", "spam"]
    assert result["train_rows"] == 4
    assert result["test_rows"] == 2
    assert result["random_state"] == 42
    assert result["train_shape"][0] == 4
    assert result["test_shape"][0] == 2
    assert result["best_model"] in {"logistic_regression", "multinomial_nb"}
    assert "logistic_regression" in result["model_scores"]
    assert "multinomial_nb" in result["model_scores"]
    assert output_dir.exists()
    assert output_dir.is_dir()
    assert (output_dir / "best_model.joblib").exists()
    assert (output_dir / "vectorizer.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_train_classifier_allows_existing_output_dir(tmp_path):
    input_csv = tmp_path / "reviews.csv"
    input_csv.write_text(
        "text,label\n"
        "Loved it,positive\n"
        "Terrible offer,negative\n"
        "Amazing support,positive\n"
        "Worst experience,negative\n"
        "Would buy again,positive\n"
        "Do not recommend,negative\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    result = train_classifier(str(input_csv), str(output_dir))

    assert result["status"] == "model_saved"
    assert result["output_dir"] == str(output_dir)
    assert result["labels"] == ["negative", "positive"]
    assert result["best_model"] in {"logistic_regression", "multinomial_nb"}


def test_train_classifier_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        train_classifier(str(tmp_path / "missing.csv"), str(tmp_path / "artifacts"))


def test_train_classifier_raises_for_missing_required_columns(tmp_path):
    input_csv = tmp_path / "bad_reviews.csv"
    input_csv.write_text(
        "message,target\nGreat product,positive\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        train_classifier(str(input_csv), str(tmp_path / "artifacts"))
