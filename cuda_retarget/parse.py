# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Parse a CUDA translation unit with Clang. Clang does not choose target spellings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import clang.cindex
from clang.cindex import Diagnostic

_LOADED = False


class ParseError(Exception):
    """Clang could not produce a token stream for this file."""


@dataclass(frozen=True)
class Tok:
    kind: str
    spelling: str
    start: int
    end: int
    # Result type of a call whose name starts at this token. Only `__hadd`
    # is recorded: the integer average and the half add share that spelling.
    call_type: str = ""


def _load_libclang() -> None:
    global _LOADED
    if _LOADED:
        return
    candidates = [
        os.environ.get("LIBCLANG_PATH", ""),
        "/usr/lib/llvm-18/lib/libclang.so.1",
        "/usr/lib/llvm-18/lib/libclang.so",
        "/usr/lib/x86_64-linux-gnu/libclang-18.so.1",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            clang.cindex.Config.set_library_file(path)
            break
    _LOADED = True


def cuda_include_dir() -> Path:
    env = os.environ.get("CUDA_INCLUDE", "")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/usr/include"),
            Path("/usr/local/cuda/include"),
            Path("/usr/lib/nvidia-cuda-toolkit/include"),
        ]
    )
    for directory in candidates:
        if (directory / "cuda_runtime.h").is_file():
            return directory
    raise ParseError(
        "cuda_runtime.h was not found. Set CUDA_INCLUDE to the CUDA header directory."
    )


def parse_file(path: Path) -> list[Tok]:
    _load_libclang()
    include = cuda_include_dir()
    index = clang.cindex.Index.create()
    tu = index.parse(
        str(path),
        args=[
            "-x",
            "cuda",
            "--cuda-host-only",
            "-std=c++17",
            "--cuda-gpu-arch=sm_70",
            f"-I{include}",
        ],
    )
    if tu is None:
        raise ParseError(f"Clang returned no translation unit for {path}")
    errors = []
    for diag in tu.diagnostics:
        if diag.severity >= Diagnostic.Error:
            errors.append(str(diag))
    if errors:
        raise ParseError("\n".join(errors[:8]))
    real = os.path.realpath(path)
    half_adds = _half_add_types(tu.cursor, real)
    tokens: list[Tok] = []
    for token in tu.get_tokens(extent=tu.cursor.extent):
        location = token.location
        if location.file is None or os.path.realpath(str(location.file)) != real:
            continue
        start = token.extent.start.offset
        tokens.append(
            Tok(
                kind=token.kind.name,
                spelling=token.spelling,
                start=start,
                end=token.extent.end.offset,
                call_type=half_adds.get(start, "") if token.spelling == "__hadd" else "",
            )
        )
    return tokens


def _half_add_types(cursor: clang.cindex.Cursor, path: str) -> dict[int, str]:
    found: dict[int, str] = {}

    def walk(node: clang.cindex.Cursor) -> None:
        loc = node.location
        if loc.file is not None and os.path.realpath(str(loc.file)) != path:
            return
        if node.kind == clang.cindex.CursorKind.CALL_EXPR and node.spelling == "__hadd":
            found[node.extent.start.offset] = node.type.spelling
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return found
