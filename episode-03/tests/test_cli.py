import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_displays_commands():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "ai-workflow-ep3" in result.stdout
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout


def test_cli_train_echoes_paths(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
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

    result = run_cli("train", str(csv_path), str(output_dir))

    assert result.returncode == 0
    assert f"Training model using: {csv_path}" in result.stdout
    assert f"Saving artifacts to:  {output_dir}" in result.stdout
    assert "Status:               model_saved" in result.stdout
    assert "Train rows:           4" in result.stdout
    assert "Test rows:            2" in result.stdout
    assert "Logistic Regression:" in result.stdout
    assert "Multinomial NB:" in result.stdout
    assert "Best model:" in result.stdout
    assert "Model file:" in result.stdout
    assert "Vectorizer file:" in result.stdout
    assert "Metrics file:" in result.stdout


def test_cli_evaluate_echoes_inputs():
    result = run_cli("evaluate", "reviews.csv", "model.joblib")

    assert result.returncode == 0
    assert "Evaluating model:     model.joblib" in result.stdout
    assert "Using dataset:        reviews.csv" in result.stdout


def test_cli_predict_echoes_input_text(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(
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
    run_cli("train", str(csv_path), str(output_dir))

    result = run_cli("predict", str(output_dir), "free prize cash")

    assert result.returncode == 0
    assert f"Model: {output_dir}" in result.stdout
    assert "Text:  free prize cash" in result.stdout
    assert "Prediction:" in result.stdout
