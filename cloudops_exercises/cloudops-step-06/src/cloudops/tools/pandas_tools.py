from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from cloudops.core.config import settings

_MAX_ROWS = 50


def _data_path(filename: str) -> Path:
    return Path(settings.data_dir) / filename


def _normalize_dt_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)


def _normalize_dt_value(value: str):
    ts = pd.to_datetime(value, utc=True)
    if hasattr(ts, "tz_localize"):
        return ts.tz_localize(None)
    return ts


def _load(filename: str) -> tuple[pd.DataFrame | None, str]:
    path = _data_path(filename)
    if not path.exists():
        return None, f"File not found: {filename}"
    return pd.read_csv(path, low_memory=False), ""


def _apply_filters(
    df: pd.DataFrame,
    filter_column: Optional[str],
    filter_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    timestamp_col: Optional[str],
) -> tuple[pd.DataFrame, str]:
    """Apply equality filter + optional date range. Returns (filtered_df, error_str)."""
    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            return df, f"Filter column '{filter_column}' not found. Available: {', '.join(df.columns)}"
        df = df[df[filter_column].astype(str) == str(filter_value)]
        if df.empty:
            return df, f"No rows matched {filter_column}={filter_value!r}."

    if (date_from or date_to):
        if not timestamp_col:
            return df, "date_from/date_to require timestamp_col to be specified."
        if timestamp_col not in df.columns:
            return df, f"timestamp_col '{timestamp_col}' not found. Available: {', '.join(df.columns)}"
        df[timestamp_col] = _normalize_dt_series(df[timestamp_col])
        if date_from:
            df = df[df[timestamp_col] >= _normalize_dt_value(date_from)]
        if date_to:
            df = df[df[timestamp_col] <= _normalize_dt_value(date_to)]
        if df.empty:
            return df, f"No rows in date range {date_from} – {date_to}."

    return df, ""


@tool
def aggregate_csv(
    filename: str,
    group_by: str,
    agg_column: str,
    agg_func: str = "count",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> str:
    """
    Group a CSV by one or more columns and aggregate a numeric column.

    To answer questions like "which 3 services had the highest average memory
    in DC-B in Q4 2024?" use filter_column='datacenter', filter_value='DC-B',
    date_from='2024-10-01', date_to='2024-12-31', timestamp_col='timestamp'.

    Args:
        filename:      CSV file name (e.g. incidents.csv)
        group_by:      Comma-separated column names to group by (e.g. 'service')
        agg_column:    Column to aggregate (e.g. 'mem_pct')
        agg_func:      One of count / mean / sum / min / max / std (default: count)
        filter_column: Optional column for equality filter (e.g. 'datacenter')
        filter_value:  Value to match in filter_column (e.g. 'DC-B')
        date_from:     Optional start datetime string (e.g. '2024-10-01')
        date_to:       Optional end datetime string (e.g. '2024-12-31')
        timestamp_col: Required when using date_from/date_to (e.g. 'timestamp')
    """
    df, err = _load(filename)
    if err:
        return err

    funcs = {"count", "mean", "sum", "min", "max", "std"}
    if agg_func not in funcs:
        return f"Unknown agg_func '{agg_func}'. Choose from: {', '.join(sorted(funcs))}"

    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]

    df, err = _apply_filters(df, filter_column, filter_value, date_from, date_to, timestamp_col)
    if err:
        return err

    # empty group_by → return a single aggregate across the whole (filtered) dataset
    if not group_cols:
        if agg_column not in df.columns:
            return f"Column '{agg_column}' not found. Available: {', '.join(df.columns)}"
        if agg_func == "count":
            return f"Total {agg_func}_{agg_column}: {len(df):,}"
        val = df[agg_column].agg(agg_func)
        return f"Total {agg_func}_{agg_column}: {round(val, 3)}"

    missing = [c for c in group_cols + [agg_column] if c not in df.columns]
    if missing:
        return f"Columns not found: {missing}. Available: {', '.join(df.columns)}"

    result = df.groupby(group_cols)[agg_column].agg(agg_func).reset_index()
    result.columns = list(group_cols) + [f"{agg_func}_{agg_column}"]
    result = result.sort_values(f"{agg_func}_{agg_column}", ascending=False)

    head = result.head(_MAX_ROWS).copy()
    head[head.select_dtypes(include="number").columns] = head.select_dtypes(include="number").round(3)
    out = head.to_string(index=False)
    suffix = f"\n\n({len(result):,} groups; showing first {min(_MAX_ROWS, len(result))})" if len(result) > _MAX_ROWS else ""
    return out + suffix


@tool
def timeseries_resample(
    filename: str,
    timestamp_col: str,
    value_col: str,
    freq: str = "1h",
    agg_func: str = "mean",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_column2: Optional[str] = None,
    filter_value2: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """
    Resample a timeseries column to a given frequency and aggregate values.

    Args:
        filename:       CSV file name (e.g. incidents.csv)
        timestamp_col:  Name of the datetime column (e.g. 'opened_at')
        value_col:      Column to aggregate (e.g. 'incident_id')
        freq:           Pandas offset alias — '1MS' monthly, '1D' daily, '1h' hourly
        agg_func:       One of mean / sum / min / max / count (default: mean)
        filter_column:  Optional column for first equality filter (e.g. 'datacenter')
        filter_value:   Value for first filter (e.g. 'DC-A')
        filter_column2: Optional column for second equality filter (e.g. 'severity')
        filter_value2:  Value for second filter (e.g. 'P0')
        date_from:      Optional start datetime string (e.g. '2024-01-01')
        date_to:        Optional end datetime string (e.g. '2024-12-31')
    """
    df, err = _load(filename)
    if err:
        return err

    for col in [timestamp_col, value_col]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {', '.join(df.columns)}"

    df, err = _apply_filters(df, filter_column, filter_value, date_from, date_to, timestamp_col)
    if err:
        return err

    # Apply optional second filter
    if filter_column2 and filter_value2:
        if filter_column2 not in df.columns:
            return f"filter_column2 '{filter_column2}' not found. Available: {', '.join(df.columns)}"
        df = df[df[filter_column2].astype(str) == str(filter_value2)]

    funcs = {"mean", "sum", "min", "max", "count"}
    if agg_func not in funcs:
        return f"Unknown agg_func '{agg_func}'. Choose from: {', '.join(sorted(funcs))}"

    df[timestamp_col] = _normalize_dt_series(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col)

    result = df[value_col].resample(freq).agg(agg_func).dropna().reset_index()
    result.columns = [timestamp_col, f"{agg_func}_{value_col}"]

    head = result.head(_MAX_ROWS).copy()
    head[head.select_dtypes(include="number").columns] = head.select_dtypes(include="number").round(3)
    out = head.to_string(index=False)
    suffix = f"\n\n({len(result):,} buckets; showing first {min(_MAX_ROWS, len(result))})" if len(result) > _MAX_ROWS else ""
    return out + suffix


@tool
def compute_correlation(
    filename_a: str,
    col_a: str,
    filename_b: str,
    col_b: str,
    join_on: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> str:
    """
    Compute Pearson correlation between two numeric columns, joining on a common key.

    Use this to check whether e.g. CPU spikes correlate with incident counts.
    Both files are joined on join_on before computing the correlation.

    Args:
        filename_a:    First CSV file
        col_a:         Numeric column from filename_a
        filename_b:    Second CSV file (may be the same file)
        col_b:         Numeric column from filename_b
        join_on:       Comma-separated column name(s) to join on (e.g. 'timestamp,host')
        date_from:     Optional start datetime string applied to both files
        date_to:       Optional end datetime string applied to both files
        timestamp_col: Required when using date_from/date_to
    """
    df_a, err = _load(filename_a)
    if err:
        return err
    df_b, err = _load(filename_b)
    if err:
        return err

    join_cols = [c.strip() for c in join_on.split(",")]

    for col in join_cols + [col_a]:
        if col not in df_a.columns:
            return f"Column '{col}' not found in {filename_a}. Available: {', '.join(df_a.columns)}"
    for col in join_cols + [col_b]:
        if col not in df_b.columns:
            return f"Column '{col}' not found in {filename_b}. Available: {', '.join(df_b.columns)}"

    df_a, err = _apply_filters(df_a, None, None, date_from, date_to, timestamp_col)
    if err:
        return err
    df_b, err = _apply_filters(df_b, None, None, date_from, date_to, timestamp_col)
    if err:
        return err

    merged = df_a[join_cols + [col_a]].merge(df_b[join_cols + [col_b]], on=join_cols, how="inner")
    if merged.empty:
        return "No rows matched after join — cannot compute correlation."

    r = merged[col_a].corr(merged[col_b])
    n = len(merged)
    direction = "positive" if r > 0 else "negative"
    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"

    return (
        f"Pearson r = {r:.3f}  (n={n:,})\n"
        f"Interpretation: {strength} {direction} correlation between {col_a} and {col_b}"
    )


@tool
def top_n(
    filename: str,
    sort_column: str,
    n: int = 10,
    ascending: bool = False,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> str:
    """
    Return the top N rows of a CSV sorted by a column.

    Args:
        filename:      CSV file name
        sort_column:   Column to sort by
        n:             Number of rows to return (default 10, max 50)
        ascending:     Sort ascending if True (default False = highest first)
        filter_column: Optional column for equality filter (e.g. 'datacenter')
        filter_value:  Value to match in filter_column (e.g. 'DC-A')
        date_from:     Optional start datetime string
        date_to:       Optional end datetime string
        timestamp_col: Required when using date_from/date_to
    """
    df, err = _load(filename)
    if err:
        return err

    if sort_column not in df.columns:
        return f"Column '{sort_column}' not found. Available: {', '.join(df.columns)}"

    df, err = _apply_filters(df, filter_column, filter_value, date_from, date_to, timestamp_col)
    if err:
        return err

    n = min(n, _MAX_ROWS)
    result = df.sort_values(sort_column, ascending=ascending).head(n)
    return result.to_string(index=False)
