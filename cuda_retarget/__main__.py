# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""cuda-retarget command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cuda_retarget.dialect import DIALECTS
from cuda_retarget.translate import TranslateError, translate_path, translate_source

_SAMPLE = """\
#include <cuda_runtime.h>
__global__ void broadcast(int* out) {
    *out = __shfl_sync(0xffffffff, threadIdx.x, 0);
}
int main() {
    int* p = nullptr;
    cudaMalloc(&p, sizeof(int));
    return 0;
}
"""


def cmd_translate(args: argparse.Namespace) -> int:
    try:
        text = translate_path(Path(args.input), args.target)
    except TranslateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    if set(DIALECTS) != {"hip", "musa", "maca"}:
        print("dialect set changed", file=sys.stderr)
        return 1
    package = Path(__file__).resolve().parent
    forbidden = ("sub" + "process", "hip" + "ify", "mus" + "ify", "cu" + "cc")
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(needle in source for needle in forbidden):
            print(f"{path.name} reaches for an official translator", file=sys.stderr)
            return 1
    try:
        hip = translate_source(_SAMPLE, "hip")
        musa = translate_source(_SAMPLE, "musa")
        maca = translate_source(_SAMPLE, "maca")
    except TranslateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if "hipMalloc" not in hip or "<hip/hip_runtime.h>" not in hip:
        print("HIP runtime rename failed", file=sys.stderr)
        return 1
    if "__shfl_sync(0xffffffff, threadIdx.x, 0, 32)" not in hip:
        print("HIP kept a 32-lane shuffle on the default wavefront width", file=sys.stderr)
        return 1
    if "musaMalloc" not in musa or "<musa_runtime.h>" not in musa:
        print("MUSA runtime rename failed", file=sys.stderr)
        return 1
    if "__shfl_sync(0xffffffff, threadIdx.x, 0);" not in musa or ", 32)" in musa:
        print("MUSA shuffle width was rewritten", file=sys.stderr)
        return 1
    if maca != _SAMPLE:
        print("MetaX dialect changed CUDA spellings", file=sys.stderr)
        return 1
    print("self-check ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuda-retarget")
    sub = parser.add_subparsers(dest="cmd", required=True)

    translate = sub.add_parser("translate")
    translate.add_argument("--target", required=True, choices=sorted(DIALECTS))
    translate.add_argument("--input", required=True)
    translate.add_argument("--output")

    sub.add_parser("self-check")

    args = parser.parse_args(argv)
    if args.cmd == "translate":
        return cmd_translate(args)
    return cmd_self_check(args)


if __name__ == "__main__":
    sys.exit(main())
