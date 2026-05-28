import re
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool

from cloudops.core.config import settings

matplotlib.use("Agg")  # non-interactive backend — no display required


def _data_path(filename: str) -> Path:
    return Path(settings.data_dir) / filename


def _charts_dir() -> Path:
    d = Path(settings.charts_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts if p)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)[:120]


def _load(filename: str) -> tuple:
    path = _data_path(filename)
    if not path.exists():
        return None, f"File not found: {filename}"
    return pd.read_csv(path, low_memory=False), ""


@tool
def plot_timeseries(
    filename: str,
    timestamp_col: str,
    value_col: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_column2: Optional[str] = None,
    filter_value2: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    freq: Optional[str] = None,
    agg_func: str = "mean",
    title: Optional[str] = None,
) -> str:
    """
    Plot a timeseries as a line chart and save it to disk.
    When freq is set (e.g. '1MS' for monthly, '1D' for daily), the series is
    resampled to that frequency (count of rows per bucket) before plotting —
    useful for incident counts per month.

    Args:
        filename:       CSV file name (e.g. incidents.csv)
        timestamp_col:  Name of the datetime column (e.g. 'opened_at')
        value_col:      Column to plot / count on the Y axis (e.g. 'incident_id')
        filter_column:  Optional first equality filter column (e.g. 'datacenter')
        filter_value:   Value for first filter (e.g. 'DC-A')
        filter_column2: Optional second equality filter column (e.g. 'severity')
        filter_value2:  Value for second filter (e.g. 'P0')
        date_from:      Optional start datetime string, e.g. '2024-01-01'
        date_to:        Optional end datetime string, e.g. '2024-12-31'
        freq:           Optional resample frequency: '1MS' monthly, '1D' daily, '1h' hourly
        agg_func:       Aggregation when freq is set: mean / sum / count (default: mean)
        title:          Optional chart title
    Returns:
        Path to the saved PNG file.
    """
    df, err = _load(filename)
    if err:
        return err

    for col in [timestamp_col, value_col]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {', '.join(df.columns)}"

    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            return f"Filter column '{filter_column}' not found."
        df = df[df[filter_column].astype(str) == str(filter_value)]

    if filter_column2 and filter_value2 is not None:
        if filter_column2 not in df.columns:
            return f"Filter column2 '{filter_column2}' not found."
        df = df[df[filter_column2].astype(str) == str(filter_value2)]

    if df.empty:
        return "No rows matched the specified filters."

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if date_from:
        df = df[df[timestamp_col] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df[timestamp_col] <= pd.to_datetime(date_to)]
    if df.empty:
        return f"No data in the specified date range ({date_from} – {date_to})."

    if freq:
        # Resample to frequency using the specified aggregation function
        valid_funcs = {"mean", "sum", "count", "min", "max"}
        func = agg_func if agg_func in valid_funcs else "mean"
        plot_df = df.set_index(timestamp_col)[value_col].resample(freq).agg(func).dropna().reset_index()
        plot_df.columns = [timestamp_col, f"{func}_{value_col}"]
        x_col, y_col = timestamp_col, f"{func}_{value_col}"
    else:
        plot_df = df
        x_col, y_col = timestamp_col, value_col

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df[x_col], plot_df[y_col], linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} over time — {filter_value or filename}")
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = _charts_dir() / f"{_slug(filename, value_col, filter_value or 'all', filter_value2 or '', freq or '')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


@tool
def plot_bar(
    filename: str,
    category_col: str,
    value_col: str,
    agg_func: str = "count",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    filter_column2: Optional[str] = None,
    filter_value2: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_col: Optional[str] = None,
    title: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """
    Bar chart: group by category_col, aggregate value_col, plot top N bars.

    Args:
        filename:       CSV file name
        category_col:   Column to group by (X axis)
        value_col:      Column to aggregate (Y axis)
        agg_func:       One of count / mean / sum / min / max (default: count)
        filter_column:  Optional column for first equality filter (e.g. 'severity')
        filter_value:   Value for first filter (e.g. 'P0')
        filter_column2: Optional column for second equality filter (e.g. 'datacenter')
        filter_value2:  Value for second filter (e.g. 'DC-A')
        date_from:      Optional start date (e.g. '2024-01-01')
        date_to:        Optional end date (e.g. '2024-12-31')
        date_col:       Datetime column to apply date_from/date_to on (e.g. 'opened_at')
        title:          Optional chart title
        top_n:          Number of bars to show (default 10)
    Returns:
        Path to the saved PNG file.
    """
    df, err = _load(filename)
    if err:
        return err

    for col in [category_col, value_col]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {', '.join(df.columns)}"

    funcs = {"count", "mean", "sum", "min", "max"}
    if agg_func not in funcs:
        return f"Unknown agg_func '{agg_func}'. Choose from: {', '.join(sorted(funcs))}"

    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            return f"filter_column '{filter_column}' not found. Available: {', '.join(df.columns)}"
        df = df[df[filter_column].astype(str) == str(filter_value)]

    if filter_column2 and filter_value2 is not None:
        if filter_column2 not in df.columns:
            return f"filter_column2 '{filter_column2}' not found. Available: {', '.join(df.columns)}"
        df = df[df[filter_column2].astype(str) == str(filter_value2)]

    if (date_from or date_to) and date_col:
        if date_col not in df.columns:
            return f"date_col '{date_col}' not found. Available: {', '.join(df.columns)}"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_from:
            df = df[df[date_col] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df[date_col] <= pd.to_datetime(date_to)]

    if df.empty:
        return "No data after applying filters."

    grouped = df.groupby(category_col)[value_col].agg(agg_func).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel(category_col)
    ax.set_ylabel(f"{agg_func}({value_col})")
    ax.set_title(title or f"{agg_func}({value_col}) by {category_col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out_path = _charts_dir() / f"{_slug(filename, category_col, value_col, agg_func, filter_value or '')}_bar.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


@tool
def plot_histogram(
    filename: str,
    value_col: str,
    bins: int = 30,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Histogram of a numeric column.

    Args:
        filename:      CSV file name
        value_col:     Numeric column to plot
        bins:          Number of histogram bins (default 30)
        filter_column: Optional column to filter on
        filter_value:  Exact value to match in filter_column
        title:         Optional chart title
    Returns:
        Path to the saved PNG file.
    """
    df, err = _load(filename)
    if err:
        return err

    if value_col not in df.columns:
        return f"Column '{value_col}' not found. Available: {', '.join(df.columns)}"

    if filter_column and filter_value is not None:
        if filter_column not in df.columns:
            return f"Filter column '{filter_column}' not found."
        df = df[df[filter_column].astype(str) == str(filter_value)]
        if df.empty:
            return f"No rows matched {filter_column}={filter_value!r}."

    series = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if series.empty:
        return f"No numeric values in column '{value_col}'."

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xlabel(value_col)
    ax.set_ylabel("count")
    ax.set_title(title or f"Distribution of {value_col} — {filter_value or filename}")
    plt.tight_layout()

    out_path = _charts_dir() / f"{_slug(filename, value_col, filter_value or 'all')}_hist.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
