"""Episode 2 EDA — data profiling functions."""

from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from jinja2 import Template
import pandas as pd


DEFAULT_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Episode 2 Data Profile</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 2rem; color: #1f2937; }
    h1, h2 { margin-bottom: 0.5rem; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.75rem; margin: 1.5rem 0; }
    .card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid #e5e7eb; padding: 0.65rem; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; }
    code { background: #eef2ff; padding: 0.1rem 0.3rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Episode 2 Data Profile</h1>
  <p>Generated at {{ report.timestamp }}</p>
  <div class=\"meta\">
    <div class=\"card\"><strong>Rows</strong><br>{{ report.rows }}</div>
    <div class=\"card\"><strong>Columns</strong><br>{{ report.columns_count }}</div>
    <div class=\"card\"><strong>Total Missing</strong><br>{{ report.missing_values_total }}</div>
    <div class=\"card\"><strong>Source</strong><br><code>{{ report.source_file }}</code></div>
  </div>

  <h2>Columns</h2>
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>Type</th>
        <th>Missing</th>
        <th>Unique</th>
        <th>Samples</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {% for column in report.column_profiles %}
      <tr>
        <td>{{ column.name }}</td>
        <td>{{ column.kind }}</td>
        <td>{{ column.missing_count }} ({{ column.missing_pct }}%)</td>
        <td>{{ column.unique_count }}</td>
        <td>{{ column.sample_values | join(', ') }}</td>
        <td>
          {% if column.numeric_summary %}
            mean={{ column.numeric_summary.mean }}, median={{ column.numeric_summary.median }}, min={{ column.numeric_summary.min }}, max={{ column.numeric_summary.max }}
          {% elif column.datetime_summary %}
            min={{ column.datetime_summary.min }}, max={{ column.datetime_summary.max }}
          {% else %}
            {{ column.top_values | map(attribute='label') | join(', ') }}
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""


def load_clean_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {csv_path}")
    return pd.read_csv(path)


def _to_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _sample_values(series: pd.Series, limit: int = 3) -> list:
    values = []
    for value in series.dropna().head(limit):
        values.append(str(_to_json_value(value)))
    return values


def _top_values(series: pd.Series, limit: int = 5) -> list:
    counts = series.dropna().astype(str).value_counts().head(limit)
    return [
        {"label": label, "count": int(count)}
        for label, count in counts.items()
    ]


def _parse_datetime_series(series: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None

    if not non_null.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        return None

    parsed = pd.Series(
        pd.to_datetime(non_null, format="%Y-%m-%d", errors="coerce"),
        index=non_null.index,
    )
    if parsed.notna().all():
        return pd.Series(
            pd.to_datetime(series.astype("string"), format="%Y-%m-%d", errors="coerce"),
            index=series.index,
        )

    return None


def _numeric_summary(series: pd.Series) -> dict:
    return {
        "min": _to_json_value(series.min()),
        "max": _to_json_value(series.max()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
    }


def _datetime_summary(series: pd.Series) -> dict:
    return {
        "min": _to_json_value(series.min()),
        "max": _to_json_value(series.max()),
    }


def _build_column_profile(name: str, series: pd.Series) -> dict:
    missing_count = int(series.isna().sum())
    total_count = int(len(series))
    missing_pct = round((missing_count / total_count) * 100, 2) if total_count else 0.0
    unique_count = int(series.nunique(dropna=True))
    profile = {
        "name": name,
        "dtype": str(series.dtype),
        "non_null_count": int(series.notna().sum()),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_count": unique_count,
        "sample_values": _sample_values(series),
        "top_values": [],
        "numeric_summary": None,
        "datetime_summary": None,
        "kind": "categorical",
    }

    if pd.api.types.is_numeric_dtype(series):
        profile["kind"] = "numeric"
        profile["numeric_summary"] = _numeric_summary(series.dropna())
        return profile

    datetime_series = _parse_datetime_series(series)
    if datetime_series is not None:
        profile["kind"] = "datetime"
        profile["datetime_summary"] = _datetime_summary(datetime_series.dropna())
        return profile

    profile["top_values"] = _top_values(series)
    return profile


def _load_template() -> Template:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "profile_report.html.j2"
    if template_path.exists():
        return Template(template_path.read_text(encoding="utf-8"))
    return Template(DEFAULT_TEMPLATE)


def generate_basic_profile(csv_path: str, output_dir: str) -> dict:
    df = load_clean_dataset(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    column_profiles = [_build_column_profile(column, df[column]) for column in df.columns]
    report_file = output_path / "basic_profile.json"
    html_report_file = output_path / "basic_profile.html"

    report = {
        "timestamp": datetime.now().isoformat(),
        "source_file": csv_path,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "numeric_columns": [profile["name"] for profile in column_profiles if profile["kind"] == "numeric"],
        "datetime_columns": [profile["name"] for profile in column_profiles if profile["kind"] == "datetime"],
        "categorical_columns": [profile["name"] for profile in column_profiles if profile["kind"] == "categorical"],
        "missing_values_total": int(df.isna().sum().sum()),
        "column_profiles": column_profiles,
        "report_file": str(report_file),
        "html_report_file": str(html_report_file),
    }

    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_report_file.write_text(_load_template().render(report=report), encoding="utf-8")
    return report