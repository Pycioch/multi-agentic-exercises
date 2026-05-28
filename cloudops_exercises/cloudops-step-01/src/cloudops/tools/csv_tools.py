from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from cloudops.core.config import settings


def _data_path(filename: str) -> Path:
    return Path(settings.data_dir) / filename


@tool
def list_csv_files() -> str:
    """List all CSV data files available in the cloud-ops dataset."""
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return "Data directory not found. Run scripts/generate_data.py to create the dataset."
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        return "No CSV files found. Run scripts/generate_data.py first."
    return "\n".join(f.name for f in files)


@tool
def describe_csv(filename: str) -> str:
    """
    Return the schema and a sample of a CSV file.

    Args:
        filename: Name of the CSV file (e.g. incidents.csv)
    """
    path = _data_path(filename)
    if not path.exists():
        return f"File not found: {filename}. Call list_csv_files to see available files."

    df = pd.read_csv(path, nrows=5)

    # counting rows without loading the full file — datasets are ≥500K rows
    row_count = sum(1 for _ in open(path)) - 1

    lines = [
        f"File: {filename}",
        f"Rows: {row_count:,}",
        f"Columns ({len(df.columns)}): {', '.join(df.columns)}",
        "",
        "Dtypes:",
        *[f"  {col}: {dtype}" for col, dtype in df.dtypes.items()],
        "",
        "Sample (5 rows):",
        df.to_string(index=False),
    ]
    return "\n".join(lines)


@tool
def query_csv(
    filename: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    columns: Optional[str] = None,
    limit: int = 20,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    covers_timestamp: Optional[str] = None,
    covers_start_col: Optional[str] = None,
    covers_end_col: Optional[str] = None,
) -> str:
    """
    Read rows from a CSV file with optional filtering.

    Args:
        filename:          CSV file name (e.g. oncall_roster.csv)
        filter_column:     Column for equality/substring filter (optional)
        filter_value:      Value to match — case-insensitive substring (optional)
        columns:           Comma-separated columns to return (optional; '*' = all)
        limit:             Max rows to return (default 20, max 100)
        date_from:         Keep rows where timestamp_col >= this value
        date_to:           Keep rows where timestamp_col <= this value
        timestamp_col:     Column used by date_from / date_to
        covers_timestamp:  Find rows where a time interval CONTAINS this moment.
                           Use to answer "who was on-call at 14:30?" —
                           e.g. covers_timestamp='2025-03-15 14:30',
                                covers_start_col='shift_start',
                                covers_end_col='shift_end'
        covers_start_col:  Start-of-interval column (required with covers_timestamp)
        covers_end_col:    End-of-interval column (required with covers_timestamp)
    """
    path = _data_path(filename)
    if not path.exists():
        return f"File not found: {filename}."

    limit = min(limit, 100)
    df = pd.read_csv(path, low_memory=False)

    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            return f"Column '{filter_column}' not found. Available: {', '.join(df.columns)}"
        mask = df[filter_column].astype(str).str.contains(
            str(filter_value), case=False, na=False
        )
        df = df[mask]

    if date_from or date_to:
        if not timestamp_col:
            return "date_from/date_to require timestamp_col to be specified."
        if timestamp_col not in df.columns:
            return f"timestamp_col '{timestamp_col}' not found. Available: {', '.join(df.columns)}"
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if date_from:
            df = df[df[timestamp_col] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df[timestamp_col] <= pd.to_datetime(date_to)]

    if covers_timestamp:
        if not covers_start_col or not covers_end_col:
            return "covers_timestamp requires both covers_start_col and covers_end_col."
        for col in (covers_start_col, covers_end_col):
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
        ts = pd.to_datetime(covers_timestamp)
        df[covers_start_col] = pd.to_datetime(df[covers_start_col], errors="coerce")
        df[covers_end_col]   = pd.to_datetime(df[covers_end_col],   errors="coerce")
        df = df[(df[covers_start_col] <= ts) & (df[covers_end_col] > ts)]

    if columns and columns.strip() != "*":
        requested = [c.strip() for c in columns.split(",")]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}. Available: {', '.join(df.columns)}"
        df = df[requested]

    if df.empty:
        return "No rows matched the filter."

    result = df.head(limit).to_string(index=False)
    total = len(df)
    suffix = f"\n\n({total:,} rows matched; showing first {min(limit, total)})" if total > limit else ""
    return result + suffix
