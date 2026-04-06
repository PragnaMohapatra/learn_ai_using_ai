import json
import subprocess
import sys
from pathlib import Path


def test_cli_help_displays_analyze_command():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "analyze", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ai-workflow-ep2" in result.stdout
    assert "analyze" in result.stdout
    assert "--charts" in result.stdout
    assert "--ai-summary" in result.stdout


def test_cli_analyze_generates_report(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,city,signup_date,purchase_amount\nAlice,30,Paris,2025-01-01,100.5\nBob,,Berlin,2025-02-15,\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "analyze",
            str(csv_path),
            str(output_dir),
            "--charts",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Analysis complete!" in result.stdout
    assert "Rows:            2" in result.stdout
    assert "Date columns:    1" in result.stdout
    assert "Charts:          enabled" in result.stdout
    assert "JSON Report:" not in result.stdout
    assert "JSON artifact:   written" in result.stdout

    report_file = output_dir / "basic_profile.json"
    html_report_file = output_dir / "basic_profile.html"
    assert report_file.exists()
    assert html_report_file.exists()

    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["columns_count"] == 5
    assert report["numeric_columns"] == ["age", "purchase_amount"]
    assert report["datetime_columns"] == ["signup_date"]
    assert report["charts"]["missing_values"]


def test_cli_ai_summary_skips_without_api_key(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,city\nAlice,30,Paris\nBob,,Berlin\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "analyze",
            str(csv_path),
            str(output_dir),
            "--ai-summary",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(Path(sys.executable).parent)},
    )

    assert result.returncode == 0
    assert "AI summary:      skipped" in result.stdout
    assert "OPENAI_API_KEY is not set." in result.stdout

    report = json.loads((output_dir / "basic_profile.json").read_text(encoding="utf-8"))
    assert report["ai_narrative"]["status"] == "skipped"