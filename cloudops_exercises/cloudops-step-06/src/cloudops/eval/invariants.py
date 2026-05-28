"""Load and expose typed invariants from data/invariants.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parents[3]  # cloudops-step-04/


@dataclass
class Invariant:
    id: str
    tier: str          # "l1" | "l2" | "l3"
    question: str
    answer: Any        # str | int | float
    source_files: list[str] = field(default_factory=list)
    note: str = ""


def load(yaml_path: Path | None = None) -> dict[str, list[Invariant]]:
    """Return {"l1": [...], "l2": [...], "l3": [...]}."""
    path = yaml_path or (_ROOT / "data" / "invariants.yaml")
    raw = yaml.safe_load(path.read_text())

    result: dict[str, list[Invariant]] = {"l1": [], "l2": [], "l3": []}
    for tier in ("l1", "l2", "l3"):
        for item in raw.get(tier, []):
            sources = item.get("source_files") or (
                [item["source_file"]] if "source_file" in item else []
            )
            result[tier].append(
                Invariant(
                    id=item["id"],
                    tier=tier,
                    question=item["question"].strip(),
                    answer=item["answer"],
                    source_files=sources,
                    note=item.get("note", ""),
                )
            )
    return result


def by_id(invariant_id: str, yaml_path: Path | None = None) -> Invariant:
    all_inv = load(yaml_path)
    for invs in all_inv.values():
        for inv in invs:
            if inv.id == invariant_id:
                return inv
    raise KeyError(f"Invariant {invariant_id!r} not found")
