import os
from pathlib import Path

import pandas as pd
import pytest

# Point config at a temp data dir before importing tools
os.environ.setdefault("DATA_DIR", "/tmp/cloudops_test_data")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def sample_csv(tmp_path, monkeypatch):
    """Write a small incidents.csv and point the tools at tmp_path."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    # Reload settings so DATA_DIR is picked up
    import importlib
    import cloudops.core.config as cfg_mod
    importlib.reload(cfg_mod)
    import cloudops.tools.csv_tools as tools_mod
    importlib.reload(tools_mod)

    df = pd.DataFrame({
        "incident_id": ["INC001", "INC002", "INC003"],
        "severity":    ["P0",     "P1",     "P2"],
        "datacenter":  ["DC-A",   "DC-B",   "DC-A"],
        "service":     ["api",    "db",     "api"],
        "timestamp":   ["2025-03-15 14:22", "2025-03-15 14:30", "2025-03-16 09:00"],
    })
    df.to_csv(tmp_path / "incidents.csv", index=False)

    yield tmp_path, tools_mod


def test_list_csv_files(sample_csv):
    _, tools = sample_csv
    result = tools.list_csv_files.invoke({})
    assert "incidents.csv" in result


def test_describe_csv_columns(sample_csv):
    _, tools = sample_csv
    result = tools.describe_csv.invoke({"filename": "incidents.csv"})
    assert "incident_id" in result
    assert "severity" in result
    assert "Rows: 3" in result


def test_describe_csv_missing_file(sample_csv):
    _, tools = sample_csv
    result = tools.describe_csv.invoke({"filename": "nonexistent.csv"})
    assert "not found" in result.lower()


def test_query_csv_no_filter(sample_csv):
    _, tools = sample_csv
    result = tools.query_csv.invoke({"filename": "incidents.csv"})
    assert "INC001" in result
    assert "INC002" in result


def test_query_csv_with_filter(sample_csv):
    _, tools = sample_csv
    result = tools.query_csv.invoke({
        "filename": "incidents.csv",
        "filter_column": "datacenter",
        "filter_value": "DC-A",
    })
    assert "INC001" in result
    assert "INC003" in result
    assert "INC002" not in result


def test_query_csv_column_selection(sample_csv):
    _, tools = sample_csv
    result = tools.query_csv.invoke({
        "filename": "incidents.csv",
        "columns": "incident_id,severity",
    })
    assert "incident_id" in result
    assert "datacenter" not in result


def test_query_csv_bad_column(sample_csv):
    _, tools = sample_csv
    result = tools.query_csv.invoke({
        "filename": "incidents.csv",
        "filter_column": "no_such_col",
        "filter_value": "x",
    })
    assert "not found" in result.lower()


def test_query_csv_limit(sample_csv):
    _, tools = sample_csv
    result = tools.query_csv.invoke({"filename": "incidents.csv", "limit": 1})
    assert "INC001" in result
    assert "INC002" not in result
