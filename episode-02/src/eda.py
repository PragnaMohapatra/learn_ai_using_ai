"""Episode 2 EDA — data profiling functions."""

import base64
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from jinja2 import Template
import matplotlib
from matplotlib.figure import Figure
import pandas as pd
from openai import OpenAI

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def build_ai_summary_prompt(report: dict[str, Any]) -> str:
    column_lines = []
    for column in report["column_profiles"]:
        details = []
        if column["numeric_summary"]:
            details.append(
                "mean={mean}, median={median}, min={min}, max={max}".format(
                    **column["numeric_summary"]
                )
            )
        elif column["datetime_summary"]:
            details.append(
                "min={min}, max={max}".format(**column["datetime_summary"])
            )
        elif column["top_values"]:
            details.append(
                "top_values=" + ", ".join(
                    f"{item['label']} ({item['count']})" for item in column["top_values"]
                )
            )
        column_lines.append(
            "- {name}: kind={kind}, missing={missing_count} ({missing_pct}%), unique={unique_count}"
            "{suffix}".format(
                name=column["name"],
                kind=column["kind"],
                missing_count=column["missing_count"],
                missing_pct=column["missing_pct"],
                unique_count=column["unique_count"],
                suffix=f", {'; '.join(details)}" if details else "",
            )
        )

    return "\n".join(
        [
            "Write a concise data-analysis narrative for this dataset.",
            "Return 3 short paragraphs:",
            "1. overall dataset shape and data quality,",
            "2. the most important numeric and categorical patterns,",
            "3. recommended next AI or ML steps.",
            "Do not invent facts beyond the profile.",
            "",
            f"Rows: {report['rows']}",
            f"Columns: {report['columns_count']}",
            f"Numeric columns: {', '.join(report['numeric_columns']) or 'none'}",
            f"Datetime columns: {', '.join(report['datetime_columns']) or 'none'}",
            f"Categorical columns: {', '.join(report['categorical_columns']) or 'none'}",
            f"Missing values total: {report['missing_values_total']}",
            "",
            "Column details:",
            *column_lines,
        ]
    )


def _log_openai_prompt(*, model: str, system_message: str, prompt: str) -> None:
    log_path = Path(
        os.getenv(
            "OPENAI_PROMPT_LOG_FILE",
            str(Path(__file__).resolve().parents[1] / "logs" / "openai_prompts.log"),
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    log_entry = (
        "\n".join(
            [
                f"[{timestamp}] model={model}",
                "[SYSTEM]",
                system_message,
                "[USER]",
                prompt,
                "-" * 80,
            ]
        )
        + "\n"
    )
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + log_entry if log_path.exists() else log_entry,
        encoding="utf-8",
    )


def generate_ai_narrative(
    report: dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
) -> dict[str, Any]:
    load_dotenv()

    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    prompt = build_ai_summary_prompt(report)
    system_message = "You are a careful data analyst. Base your summary only on the provided dataset profile."

    if not resolved_api_key and client is None:
        return {
            "status": "skipped",
            "provider": "openai",
            "model": resolved_model,
            "content": None,
            "reason": "OPENAI_API_KEY is not set.",
            "prompt": prompt,
        }

    try:
        active_client = client or OpenAI(api_key=resolved_api_key)
        _log_openai_prompt(model=resolved_model, system_message=system_message, prompt=prompt)
        if hasattr(active_client, "responses"):
            response = active_client.responses.create(
                model=resolved_model,
                input=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=280,
            )
            content = response.output_text.strip()
        else:
            # Backward-compatible path for older OpenAI SDK versions.
            response = active_client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=280,
            )
            content = (response.choices[0].message.content or "").strip()
        return {
            "status": "generated",
            "provider": "openai",
            "model": resolved_model,
            "content": content,
            "reason": None,
            "prompt": prompt,
        }
    except Exception as exc:
        return {
            "status": "error",
            "provider": "openai",
            "model": resolved_model,
            "content": None,
            "reason": str(exc),
            "prompt": prompt,
        }


def _fig_to_base64(fig: Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=96)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_charts(df: pd.DataFrame, column_profiles: list) -> dict:
    charts = {}

    # Missing values bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#b45309")
        ax.set_title("Missing Values per Column")
        ax.set_xlabel("Count")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Missing Values")
    ax.spines[["top", "right"]].set_visible(False)
    charts["missing_values"] = _fig_to_base64(fig)

    # Numeric histograms
    charts["numeric"] = {}
    for profile in column_profiles:
        if profile["kind"] != "numeric":
            continue
        col = profile["name"]
        fig, ax = plt.subplots(figsize=(6, 3))
        df[col].dropna().plot(kind="hist", ax=ax, bins=10, color="#2563eb", edgecolor="white")
        ax.set_title(f"{col} — distribution")
        ax.set_xlabel(col)
        ax.spines[["top", "right"]].set_visible(False)
        charts["numeric"][col] = _fig_to_base64(fig)

    # Categorical bar charts
    charts["categorical"] = {}
    for profile in column_profiles:
        if profile["kind"] != "categorical" or not profile["top_values"]:
            continue
        col = profile["name"]
        labels = [item["label"] for item in profile["top_values"]]
        counts = [item["count"] for item in profile["top_values"]]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(labels[::-1], counts[::-1], color="#16a34a")
        ax.set_title(f"{col} — top values")
        ax.set_xlabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        charts["categorical"][col] = _fig_to_base64(fig)

    return charts

def generate_basic_profile(
    csv_path: str,
    output_dir: str,
    with_charts: bool = False,
    with_ai_summary: bool = False,
    ai_client: Optional[Any] = None,
) -> dict:
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
        "charts": {},
        "ai_narrative": None,
    }

    if with_charts:
        report["charts"] = generate_charts(df, column_profiles)

    if with_ai_summary:
        report["ai_narrative"] = generate_ai_narrative(report, client=ai_client)

    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_report_file.write_text(_load_template().render(report=report), encoding="utf-8")
    return report