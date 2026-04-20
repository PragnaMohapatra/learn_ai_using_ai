from src import predict_text, train_classifier

import pytest


def _train_fixture(tmp_path):
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
    train_classifier(str(input_csv), str(output_dir))
    return output_dir


def test_predict_text_returns_label(tmp_path):
    output_dir = _train_fixture(tmp_path)

    result = predict_text(str(output_dir), "free prize cash")

    assert result["model_dir"] == str(output_dir)
    assert result["text"] == "free prize cash"
    assert result["prediction"] in {"ham", "spam"}


def test_predict_text_raises_for_empty_text(tmp_path):
    output_dir = _train_fixture(tmp_path)

    with pytest.raises(ValueError, match="must not be empty"):
        predict_text(str(output_dir), "   ")


def test_predict_text_raises_when_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        predict_text(str(tmp_path), "hello")
