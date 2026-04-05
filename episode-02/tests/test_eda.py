import json

import pytest

from src.eda import generate_basic_profile, load_clean_dataset


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
