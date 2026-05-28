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
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Plot a timeseries column as a line chart and save it to disk.

    Args:
        filename:      CSV file name (e.g. metrics_cpu.csv)
        timestamp_col: Name of the datetime column
        value_col:     Numeric column to plot on the Y axis
        filter_column: Optional column to filter on (e.g. 'host')
        filter_value:  Exact value to match in filter_column (e.g. 'dc-a-host-02')
        date_from:     Optional start datetime string, e.g. '2025-03-15'
        date_to:       Optional end datetime string, e.g. '2025-03-15 23:59'
        title:         Optional chart title
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
        if df.empty:
            return f"No rows matched {filter_column}={filter_value!r}."

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if date_from:
        df = df[df[timestamp_col] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df[timestamp_col] <= pd.to_datetime(date_to)]
    if df.empty:
        return f"No data in the specified date range ({date_from} – {date_to})."

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[timestamp_col], df[value_col], linewidth=0.8)
    ax.set_xlabel(timestamp_col)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} over time — {filter_value or filename}")
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = _charts_dir() / f"{_slug(filename, value_col, filter_value or 'all')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


@tool
def plot_bar(
    filename: str,
    category_col: str,
    value_col: str,
    agg_func: str = "count",
    title: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """
    Bar chart: group by category_col, aggregate value_col, plot top N bars.

    Args:
        filename:     CSV file name
        category_col: Column to group by (X axis)
        value_col:    Column to aggregate (Y axis)
        agg_func:     One of count / mean / sum / min / max (default: count)
        title:        Optional chart title
        top_n:        Number of bars to show (default 10)
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

    grouped = df.groupby(category_col)[value_col].agg(agg_func).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel(category_col)
    ax.set_ylabel(f"{agg_func}({value_col})")
    ax.set_title(title or f"{agg_func}({value_col}) by {category_col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out_path = _charts_dir() / f"{_slug(filename, category_col, value_col, agg_func)}_bar.png"
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
