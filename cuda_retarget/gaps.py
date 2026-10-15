# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Source replacements that are allowed only after an official tool has failed."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "gaps" / "registry.json"

# Statuses cuda-census can record for an official toolchain. A gap may cite
# only a failure. A pass is not a reason to rewrite the official output.
FAILURES = ("compile-fail", "run-fail", "numeric-mismatch", "unparsed-stdout")


class GapError(Exception):
    """The gap registry breaks the citation rule."""


def load_registry(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRY).read_text(encoding="utf-8"))


def validate(registry: dict) -> list[str]:
    errors: list[str] = []
    gaps = registry.get("gaps")
    if not isinstance(gaps, list):
        return ["gaps must be a list"]
    for index, gap in enumerate(gaps):
        label = str(gap.get("id", index))
        if not gap.get("admit"):
            continue
        if not gap.get("census_case"):
            errors.append(f"{label}: an admitted gap needs census_case from cuda-census")
        if gap.get("census_status") not in FAILURES:
            errors.append(
                f"{label}: census_status must be one of {', '.join(FAILURES)}"
            )
        if gap.get("target") not in ("hip", "musa", "maca"):
            errors.append(f"{label}: target must be hip, musa, or maca")
        if not gap.get("find"):
            errors.append(f"{label}: find must be a non-empty literal")
        if "replace" not in gap:
            errors.append(f"{label}: replace must be present")
    return errors


def apply_admitted(text: str, registry: dict, target: str) -> str:
    errors = validate(registry)
    if errors:
        raise GapError("; ".join(errors))
    for gap in registry["gaps"]:
        if not gap.get("admit") or gap.get("target") != target:
            continue
        needle = gap["find"]
        if needle not in text:
            raise GapError(
                f"{gap.get('id', needle)}: official output does not contain the cited text"
            )
        text = text.replace(needle, gap["replace"], 1)
    return text
