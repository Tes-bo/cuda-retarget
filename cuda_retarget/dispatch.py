# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Build and run the official-tool command lines in targets.json."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from cuda_retarget.gaps import REGISTRY, GapError, apply_admitted, load_registry, validate

ROOT = Path(__file__).resolve().parents[1]


class DispatchError(Exception):
    """The request is not something the official tools can be asked to do."""


def load_targets() -> dict:
    return json.loads((ROOT / "targets.json").read_text(encoding="utf-8"))


def slots_for(source: Path, output_dir: Path) -> dict[str, str]:
    return {
        "source": str(source),
        "translated": str(output_dir / "translated.cu"),
        "copied": str(output_dir / "main.cu"),
        "binary": str(output_dir / "case"),
    }


def render(argv: list[str], slots: dict[str, str]) -> list[str]:
    out = []
    for part in argv:
        text = part
        for key, value in slots.items():
            text = text.replace("{" + key + "}", value)
        out.append(text)
    return out


def plan(target: str, source: Path, output_dir: Path) -> list[dict]:
    targets = load_targets()
    if target not in targets:
        raise DispatchError(f"unknown target {target}")
    slots = slots_for(source, output_dir)
    steps = []
    for step in targets[target]["steps"]:
        item = {"id": step["id"]}
        if "argv" in step:
            item["argv"] = render(step["argv"], slots)
        if "stdout" in step:
            item["stdout"] = step["stdout"].format(**slots)
        if "from" in step:
            item["from"] = step["from"].format(**slots)
            item["to"] = step["to"].format(**slots)
        if "input" in step:
            item["input"] = step["input"].format(**slots)
        steps.append(item)
    return steps


def execute(
    target: str,
    source: Path,
    output_dir: Path,
    registry_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    if not source.is_file():
        raise DispatchError(f"no such source {source}")
    registry = load_registry(registry_path or REGISTRY)
    problems = validate(registry)
    if problems:
        raise GapError("; ".join(problems))
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = plan(target, source, output_dir)
    log = []
    for step in steps:
        if step["id"] == "copy":
            log.append(step)
            if not dry_run:
                Path(step["to"]).write_bytes(Path(step["from"]).read_bytes())
            continue
        if step["id"] == "gaps":
            log.append({"id": "gaps", "input": step["input"]})
            if dry_run:
                continue
            path = Path(step["input"])
            text = path.read_text(encoding="utf-8")
            path.write_text(apply_admitted(text, registry, target), encoding="utf-8")
            continue
        argv = step["argv"]
        resolved = shutil.which(argv[0])
        record = {"id": step["id"], "argv": argv, "on_path": resolved is not None}
        log.append(record)
        if dry_run:
            continue
        if resolved is None:
            raise DispatchError(f"not on PATH: {argv[0]}")
        proc = subprocess.run(
            [resolved, *argv[1:]],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        record["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise DispatchError(
                f"{argv[0]} exited {proc.returncode}\n{proc.stderr[-4000:]}"
            )
        if "stdout" in step:
            Path(step["stdout"]).write_text(proc.stdout, encoding="utf-8")
    return {"target": target, "dry_run": dry_run, "steps": log}
