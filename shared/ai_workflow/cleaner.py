"""Shared data cleaning utilities reused across episodes."""

from datetime import datetime
from pathlib import Path

import pandas as pd


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers and replace spaces/special chars with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows and return the cleaned frame plus dropped count."""
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> tuple[pd.DataFrame, int]:
    """Handle missing values by dropping rows or filling defaults."""
    missing_before = int(df.isnull().sum().sum())

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        for col in df.columns:
            if df[col].dtype in ("float64", "int64"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("UNKNOWN")

    return df, missing_before


def generate_report(
    original_rows: int,
    cleaned_rows: int,
    duplicates_dropped: int,
    missing_handled: int,
    output_path: str,
) -> dict:
    """Generate a summary report of the cleaning operation."""
    return {
        "timestamp": datetime.now().isoformat(),
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_dropped": duplicates_dropped,
        "missing_values_handled": missing_handled,
        "output_file": output_path,
        "rows_removed_total": original_rows - cleaned_rows,
    }


def clean_csv(input_path: str, output_path: str, missing_strategy: str = "drop") -> dict:
    """Run the full CSV cleaning pipeline and return a report."""
    df = pd.read_csv(input_path)
    original_rows = len(df)

    df = clean_headers(df)
    df = trim_strings(df)
    df, duplicates_dropped = drop_duplicates(df)
    df, missing_handled = handle_missing(df, strategy=missing_strategy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return generate_report(
        original_rows=original_rows,
        cleaned_rows=len(df),
        duplicates_dropped=duplicates_dropped,
        missing_handled=missing_handled,
        output_path=output_path,
    )
