import json

import pytest

from src.eda import generate_ai_narrative, generate_basic_profile, load_clean_dataset


def test_load_clean_dataset_reads_csv(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,purchase_amount\nAlice,30,100.5\nBob,45,\n",
        encoding="utf-8",
    )

    dataframe = load_clean_dataset(str(csv_path))

    assert list(dataframe.columns) == ["name", "age", "purchase_amount"]
    assert dataframe.shape == (2, 3)
    assert dataframe.loc[0, "name"] == "Alice"


def test_load_clean_dataset_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Clean dataset not found"):
        load_clean_dataset(str(missing_path))


def test_generate_basic_profile_writes_report(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,city,signup_date,purchase_amount\nAlice,30,Paris,2025-01-01,100.5\nBob,,Berlin,2025-02-15,\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    report = generate_basic_profile(str(csv_path), str(output_dir))

    report_file = output_dir / "basic_profile.json"
    html_report_file = output_dir / "basic_profile.html"
    assert report["rows"] == 2
    assert report["columns_count"] == 5
    assert report["columns"] == ["name", "age", "city", "signup_date", "purchase_amount"]
    assert report["numeric_columns"] == ["age", "purchase_amount"]
    assert report["datetime_columns"] == ["signup_date"]
    assert report["categorical_columns"] == ["name", "city"]
    assert report["missing_values_total"] == 2
    assert report["report_file"] == str(report_file)
    assert report["html_report_file"] == str(html_report_file)
    assert report["charts"] == {}
    assert report_file.exists()
    assert html_report_file.exists()

    signup_profile = next(
        profile for profile in report["column_profiles"] if profile["name"] == "signup_date"
    )
    assert signup_profile["kind"] == "datetime"
    assert signup_profile["datetime_summary"]["min"] == "2025-01-01T00:00:00"

    city_profile = next(
        profile for profile in report["column_profiles"] if profile["name"] == "city"
    )
    assert city_profile["kind"] == "categorical"
    assert city_profile["top_values"][0]["label"] == "Paris"

    written_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert written_report == report

    html_report = html_report_file.read_text(encoding="utf-8")
    assert "Episode 2 Data Profile" in html_report
    assert "signup_date" in html_report


def test_generate_basic_profile_with_charts(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,city,signup_date,purchase_amount\nAlice,30,Paris,2025-01-01,100.5\nBob,,Berlin,2025-02-15,80.0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    report = generate_basic_profile(
        str(csv_path),
        str(output_dir),
        with_charts=True,
    )

    assert report["charts"]["missing_values"]
    assert report["charts"]["numeric"]["age"]
    assert report["charts"]["categorical"]["city"]

    html_report = (output_dir / "basic_profile.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_report


def test_generate_basic_profile_with_ai_summary(tmp_path, monkeypatch):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age,city\nAlice,30,Paris\nBob,,Berlin\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    prompt_log_file = tmp_path / "openai_prompts.log"
    monkeypatch.setenv("OPENAI_PROMPT_LOG_FILE", str(prompt_log_file))

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            class Response:
                output_text = "Clean dataset, mostly complete. Age ranges 30. Consider segmentation."
            return Response()

    class FakeClient:
        responses = FakeResponses()

    report = generate_basic_profile(
        str(csv_path),
        str(output_dir),
        with_ai_summary=True,
        ai_client=FakeClient(),
    )

    assert report["ai_narrative"]["status"] == "generated"
    assert "segmentation" in report["ai_narrative"]["content"]
    assert report["ai_narrative"]["reason"] is None

    html_report = (output_dir / "basic_profile.html").read_text(encoding="utf-8")
    assert "AI Narrative" in html_report
    assert "segmentation" in html_report
    assert "Prompt Used" in html_report
    assert "Write a concise data-analysis narrative for this dataset." in html_report
    assert prompt_log_file.exists()

    prompt_log = prompt_log_file.read_text(encoding="utf-8")
    assert "model=gpt-4o-mini" in prompt_log
    assert "[SYSTEM]" in prompt_log
    assert "[USER]" in prompt_log
    assert "Rows: 2" in prompt_log


def test_generate_basic_profile_ai_summary_off_by_default(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,age\nAlice,30\nBob,45\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    report = generate_basic_profile(str(csv_path), str(output_dir))

    assert report["ai_narrative"] is None
    html_report = (output_dir / "basic_profile.html").read_text(encoding="utf-8")
    assert "AI Narrative" not in html_report


def test_generate_ai_narrative_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    narrative = generate_ai_narrative(
        {
            "rows": 2,
            "columns_count": 2,
            "numeric_columns": ["age"],
            "datetime_columns": [],
            "categorical_columns": ["city"],
            "missing_values_total": 0,
            "column_profiles": [
                {
                    "name": "age",
                    "kind": "numeric",
                    "missing_count": 0,
                    "missing_pct": 0.0,
                    "unique_count": 2,
                    "numeric_summary": {"mean": 37.5, "median": 37.5, "min": 30, "max": 45},
                    "datetime_summary": None,
                    "top_values": [],
                }
            ],
        }
    )

    assert narrative["status"] == "skipped"
    assert narrative["content"] is None
    assert narrative["reason"] == "OPENAI_API_KEY is not set."
