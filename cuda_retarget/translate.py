# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Rewrite CUDA source for one dialect.

Edits are spans over Clang tokens. Comments and string literals are not
identifier tokens, so a CUDA name inside them stays. Macro bodies are tokenised
too, so a name inside `#define` is rewritten at its definition.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cuda_retarget.dialect import (
    BITCAST_HELPERS,
    BYTE_PERM_HELPER,
    DIALECTS,
    DOUBLE_WORD_HELPERS,
    DP4A_HELPER,
    FUNNEL_HELPERS,
    HADD_HELPERS,
    MULHI_HELPERS,
    SAD_HELPERS,
    MATCH_ALL_HELPER,
    PACKED_ABS_HELPERS,
    PACKED_HELPERS,
    PACKED_MINMAX_HELPERS,
    PACKED_AVG_HELPERS,
    PACKED_CMP_HELPERS,
    PACKED_SAD_HELPERS,
    PACKED_SET_HELPERS,
    BIT_HELPERS,
    MUL24_HELPERS,
    MATCH_ANY_HELPER,
    VOTE_HELPERS,
    D2F_CORE,
    ROUND_BITS,
    ROUND_HELPERS,
    ROUND_I32,
    ROUND_I64,
    SATURATE_HELPERS,
    FARITH_ADD,
    FARITH_FMA,
    FARITH_HELPERS,
    FARITH_MUL,
    FARITH_PACK,
    Dialect,
    farith_needs_add,
    farith_needs_fma,
    farith_needs_mul,
    round_needs_bits,
    round_needs_d2f,
    round_needs_i32,
    round_needs_i64,
    runtime_name,
)
from cuda_retarget.parse import ParseError, Tok, parse_file

_FULL_MASKS = {"0xffffffff", "0xffffffffu", "0xffffffffl", "0xffffffffll", "0xffffffffull"}
# Same default width as __shfl_sync: the last argument defaults to warpSize.
_SHUFFLES = {"__shfl_sync", "__shfl_xor_sync", "__shfl_up_sync", "__shfl_down_sync"}
_MATCHES = {"__match_any_sync": 2, "__match_all_sync": 3}
_VOTES = {
    "__ballot_sync": "cts_ballot_sync",
    "__any_sync": "cts_any_sync",
    "__all_sync": "cts_all_sync",
    "__uni_sync": "cts_uni_sync",
}


class TranslateError(Exception):
    """The source is CUDA that this rewriter will not pretend to understand."""


def translate_path(path: Path, target: str) -> str:
    text = path.read_text(encoding="utf-8")
    return translate_source(text, target, filename=path.name)


def translate_source(text: str, target: str, filename: str = "input.cu") -> str:
    if target not in DIALECTS:
        raise TranslateError(f"unknown target {target}")
    dialect = DIALECTS[target]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        path.write_text(text, encoding="utf-8")
        try:
            tokens = parse_file(path)
        except ParseError as exc:
            raise TranslateError(str(exc)) from exc
    edits = _edits(text, tokens, dialect)
    rewritten = _apply(text, edits)
    helpers = _helpers(dialect, rewritten.decode())
    if not helpers:
        return rewritten.decode()
    insert_at = _map_offset(_include_end(tokens), edits)
    block = ("\n" + helpers + "\n").encode()
    return (rewritten[:insert_at] + block + rewritten[insert_at:]).decode()


def _edits(text: str, tokens: list[Tok], dialect: Dialect) -> list[tuple[int, int, str]]:
    edits: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        header = _header_edit(tokens, index, dialect)
        if header is not None:
            edits.append(header[0])
            index = header[1]
            continue
        call = _call_edit(text, tokens, index, dialect)
        if call is not None:
            edits.append(call)
        token = tokens[index]
        if token.kind == "IDENTIFIER":
            if (
                dialect.shuffle_width_32
                and token.spelling == "warpSize"
                and not _is_define_name(tokens, index)
            ):
                # CUDA's builtin is 32. On AMD the same name follows the wavefront.
                edits.append((token.start, token.end, "32"))
            else:
                renamed = runtime_name(token.spelling, dialect)
                if renamed is not None:
                    edits.append((token.start, token.end, renamed))
        index += 1
    _reject_overlaps(edits)
    return edits


def _header_edit(
    tokens: list[Tok], index: int, dialect: Dialect
) -> tuple[tuple[int, int, str], int] | None:
    window = tokens[index : index + 5]
    if len(window) < 5:
        return None
    if [token.spelling for token in window] != ["<", "cuda_runtime", ".", "h", ">"]:
        return None
    name = "cuda_runtime.h"
    if name not in dialect.headers:
        return None
    return (window[0].start, window[-1].end, f"<{dialect.headers[name]}>"), index + 5


def _call_edit(
    text: str, tokens: list[Tok], index: int, dialect: Dialect
) -> tuple[int, int, str] | None:
    token = tokens[index]
    if token.kind != "IDENTIFIER":
        return None
    if index + 1 >= len(tokens) or tokens[index + 1].spelling != "(":
        return None
    if token.spelling in _SHUFFLES and dialect.shuffle_width_32:
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            raise TranslateError(f"{token.spelling} arguments are not a simple call")
        args, close = parsed
        if len(args) == 3 and _is_full_mask(text, args[0]):
            # Insert before the closing parenthesis. CUDA's omitted width is 32.
            # The identifier loop still rewrites runtime names inside the call.
            paren = tokens[close].start
            return (paren, paren, ", 32")
    if dialect.match_width_32 and token.spelling in _MATCHES:
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            raise TranslateError(f"{token.spelling} arguments are not a simple call")
        args, _close = parsed
        if len(args) == _MATCHES[token.spelling] and _is_full_mask(text, args[0]):
            replacement = "cts_match_any_sync" if token.spelling == "__match_any_sync" else "cts_match_all_sync"
            return (token.start, token.end, replacement)
    if dialect.vote_width_32 and token.spelling in _VOTES:
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            raise TranslateError(f"{token.spelling} arguments are not a simple call")
        args, _close = parsed
        if len(args) == 2 and _is_full_mask(text, args[0]):
            return (token.start, token.end, _VOTES[token.spelling])
    if dialect.syncwarp_mask_32 and token.spelling == "__syncwarp":
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            raise TranslateError("__syncwarp arguments are not a simple call")
        args, close = parsed
        if len(args) == 0:
            # CUDA's omitted mask is 0xffffffff, one warp of 32 lanes.
            return (token.start, tokens[close].end, "__syncwarp(0xffffffff)")
    if dialect.lower_bitcasts and token.spelling in BITCAST_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = BITCAST_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_bitcasts and token.spelling in DOUBLE_WORD_HELPERS:
        parsed = _arguments(tokens, index + 1)
        expected = 2 if token.spelling == "__hiloint2double" else 1
        if parsed is None or len(parsed[0]) != expected:
            raise TranslateError(f"{token.spelling} must have {expected} argument(s)")
        replacement, _helper = DOUBLE_WORD_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_funnelshifts and token.spelling in FUNNEL_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 3:
            raise TranslateError(f"{token.spelling} must have three arguments")
        replacement, _helper = FUNNEL_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_dp4a and token.spelling == "__dp4a":
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 3:
            raise TranslateError("__dp4a must have three arguments")
        return (token.start, token.end, "cts_dp4a")
    if dialect.lower_packed_arith and token.spelling in PACKED_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_minmax and token.spelling in PACKED_MINMAX_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_MINMAX_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_abs and token.spelling in PACKED_ABS_HELPERS:
        parsed = _arguments(tokens, index + 1)
        replacement, _helper, arity = PACKED_ABS_HELPERS[token.spelling]
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} argument(s)")
        return (token.start, token.end, replacement)
    if dialect.lower_byte_perm and token.spelling == "__byte_perm":
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 3:
            raise TranslateError("__byte_perm must have three arguments")
        return (token.start, token.end, "cts_byte_perm")
    if dialect.lower_hadd and token.spelling in HADD_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = HADD_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_mulhi and token.spelling in MULHI_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = MULHI_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_sad and token.spelling in SAD_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 3:
            raise TranslateError(f"{token.spelling} must have three arguments")
        replacement, _helper = SAD_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_sad and token.spelling in PACKED_SAD_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_SAD_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_cmp and token.spelling in PACKED_CMP_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_CMP_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_set and token.spelling in PACKED_SET_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_SET_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_packed_avg and token.spelling in PACKED_AVG_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = PACKED_AVG_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_bitops and token.spelling in BIT_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = BIT_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_mul24 and token.spelling in MUL24_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = MUL24_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_round and token.spelling in ROUND_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = ROUND_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_saturate and token.spelling in SATURATE_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = SATURATE_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_farith and token.spelling in FARITH_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 3 if token.spelling.startswith("__fmaf_") else 2
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = FARITH_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    return None


def _is_define_name(tokens: list[Tok], index: int) -> bool:
    return index >= 2 and tokens[index - 2].spelling == "#" and tokens[index - 1].spelling == "define"


def _arguments(tokens: list[Tok], open_index: int) -> tuple[list[tuple[int, int]], int] | None:
    if tokens[open_index].spelling != "(":
        return None
    depth = 0
    arg_start: int | None = None
    args: list[tuple[int, int]] = []
    for index in range(open_index, len(tokens)):
        spelling = tokens[index].spelling
        if spelling == "(":
            depth += 1
            if depth == 1:
                arg_start = None
            continue
        if spelling == ")":
            depth -= 1
            if depth == 0:
                if arg_start is not None:
                    args.append((arg_start, tokens[index - 1].end))
                return args, index
            continue
        if depth == 1 and spelling == ",":
            if arg_start is None:
                return None
            args.append((arg_start, tokens[index - 1].end))
            arg_start = None
            continue
        if depth == 1 and arg_start is None:
            arg_start = tokens[index].start
    return None


def _slice(text: str, start: int, end: int) -> str:
    return text.encode()[start:end].decode()


def _is_full_mask(text: str, span: tuple[int, int]) -> bool:
    spelling = _slice(text, span[0], span[1]).strip().lower()
    return spelling in _FULL_MASKS


def _apply(text: str, edits: list[tuple[int, int, str]]) -> bytes:
    data = bytearray(text.encode())
    for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        data[start:end] = replacement.encode()
    return bytes(data)


def _helpers(dialect: Dialect, rewritten: str) -> str:
    blocks = []
    if dialect.lower_bitcasts:
        for name, helper in BITCAST_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
        for name, helper in DOUBLE_WORD_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.match_width_32 and (
        "cts_match_any_sync(" in rewritten or "cts_match_all_sync(" in rewritten
    ):
        blocks.append(MATCH_ANY_HELPER.strip("\n"))
        if "cts_match_all_sync(" in rewritten:
            blocks.append(MATCH_ALL_HELPER.strip("\n"))
    if dialect.vote_width_32:
        used = [name for name in VOTE_HELPERS if f"{name}(" in rewritten]
        if used:
            blocks.append(VOTE_HELPERS["cts_ballot_sync"].strip("\n"))
            for name in used:
                if name != "cts_ballot_sync":
                    blocks.append(VOTE_HELPERS[name].strip("\n"))
    if dialect.lower_funnelshifts:
        for name, helper in FUNNEL_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_dp4a and "cts_dp4a(" in rewritten:
        blocks.append(DP4A_HELPER.strip("\n"))
    if dialect.lower_packed_arith:
        for name, helper in PACKED_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_minmax:
        for name, helper in PACKED_MINMAX_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_abs:
        for name, helper, _arity in PACKED_ABS_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_byte_perm and "cts_byte_perm(" in rewritten:
        blocks.append(BYTE_PERM_HELPER.strip("\n"))
    if dialect.lower_hadd:
        for name, helper in HADD_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_mulhi:
        used = {name for name, _helper in MULHI_HELPERS.values() if f"{name}(" in rewritten}
        if "cts_mul64hi" in used:
            used.add("cts_umul64hi")
        for name, helper in MULHI_HELPERS.values():
            if name in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_sad:
        for name, helper in SAD_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_sad:
        for name, helper in PACKED_SAD_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_cmp:
        for name, helper in PACKED_CMP_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_set:
        for name, helper in PACKED_SET_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_packed_avg:
        for name, helper in PACKED_AVG_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_bitops:
        for name, helper in BIT_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_mul24:
        for name, helper in MUL24_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_round:
        used = [(name, helper) for name, helper in ROUND_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if round_needs_bits(names):
            blocks.append(ROUND_BITS.strip("\n"))
        if round_needs_i32(names):
            blocks.append(ROUND_I32.strip("\n"))
        if round_needs_i64(names):
            blocks.append(ROUND_I64.strip("\n"))
        if round_needs_d2f(names):
            blocks.append(D2F_CORE.strip("\n"))
        for _name, helper in used:
            blocks.append(helper.strip("\n"))
    if dialect.lower_saturate:
        for name, helper in SATURATE_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_farith:
        used = [(name, helper) for name, helper in FARITH_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.append(FARITH_PACK.strip("\n"))
            if farith_needs_add(names):
                blocks.append(FARITH_ADD.strip("\n"))
            if farith_needs_mul(names):
                blocks.append(FARITH_MUL.strip("\n"))
            if farith_needs_fma(names):
                blocks.append(FARITH_FMA.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    return "\n\n".join(blocks)


def _include_end(tokens: list[Tok]) -> int:
    end = 0
    index = 0
    while index < len(tokens):
        if tokens[index].spelling == "#" and index + 1 < len(tokens) and tokens[index + 1].spelling == "include":
            scan = index + 2
            while scan < len(tokens) and tokens[scan].spelling != ">":
                scan += 1
            if scan < len(tokens):
                end = tokens[scan].end
                index = scan
        index += 1
    return end


def _map_offset(offset: int, edits: list[tuple[int, int, str]]) -> int:
    delta = 0
    for start, end, text in edits:
        if end <= offset:
            delta += len(text.encode()) - (end - start)
    return offset + delta


def _reject_overlaps(edits: list[tuple[int, int, str]]) -> None:
    ordered = sorted(edits)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise TranslateError(
                f"overlapping edits at {previous[0]}:{previous[1]} and {current[0]}:{current[1]}"
            )
