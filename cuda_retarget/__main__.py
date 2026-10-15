# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""cuda-retarget command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cuda_retarget.dispatch import DispatchError, execute, load_targets
from cuda_retarget.gaps import FAILURES, GapError, load_registry, validate


def cmd_run(args: argparse.Namespace) -> int:
    try:
        report = execute(
            args.target,
            Path(args.input),
            Path(args.output_dir),
            dry_run=args.dry_run,
        )
    except (DispatchError, GapError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


def cmd_gaps(_args: argparse.Namespace) -> int:
    registry = load_registry()
    problems = validate(registry)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    admitted = [gap for gap in registry["gaps"] if gap.get("admit")]
    print(f"{len(admitted)} admitted gap(s), {len(registry['gaps'])} registered")
    for gap in admitted:
        print(f"{gap['id']}\t{gap['target']}\t{gap['census_case']}\t{gap['census_status']}")
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    targets = load_targets()
    for name in ("hip", "musa", "maca"):
        if name not in targets:
            print(f"missing target {name}", file=sys.stderr)
            return 1
        argv0s = [
            step["argv"][0]
            for step in targets[name]["steps"]
            if "argv" in step
        ]
        if "cuda-retarget" in argv0s:
            print("a target calls cuda-retarget", file=sys.stderr)
            return 1
    if validate(load_registry()):
        print("shipped gap registry is not empty of errors", file=sys.stderr)
        return 1
    bad = validate(
        {
            "gaps": [
                {
                    "id": "uncited",
                    "admit": True,
                    "target": "hip",
                    "find": "cudaMalloc",
                    "replace": "hipMalloc",
                }
            ]
        }
    )
    if not bad:
        print("an admitted gap without census_case was accepted", file=sys.stderr)
        return 1
    cited = validate(
        {
            "gaps": [
                {
                    "id": "cited",
                    "admit": True,
                    "target": "hip",
                    "census_case": "warp_shuffle",
                    "census_status": FAILURES[0],
                    "find": "cudaMalloc",
                    "replace": "hipMalloc",
                }
            ]
        }
    )
    if cited:
        print("a cited failure was rejected: " + "; ".join(cited), file=sys.stderr)
        return 1
    print("self-check ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuda-retarget")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--target", required=True, choices=sorted(load_targets()))
    run.add_argument("--input", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--dry-run", action="store_true")

    sub.add_parser("gaps")
    sub.add_parser("self-check")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "gaps":
        return cmd_gaps(args)
    return cmd_self_check(args)


if __name__ == "__main__":
    sys.exit(main())
