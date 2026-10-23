# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Rewrite CUDA source for one dialect.

Edits are spans over Clang tokens. Comments and string literals are not
identifier tokens, so a CUDA name inside them stays. Macro bodies are tokenised
too, so a name inside `#define` is rewritten at its definition.
"""

from __future__ import annotations

import re
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
    I2D_S64,
    I2D_U64,
    ROUND_I32,
    ROUND_I64,
    SATURATE_HELPERS,
    DDIV_DIV,
    DDIV_HELPERS,
    DDIV_SQRT,
    FDIV_DIV,
    FDIV_HELPERS,
    FDIV_ISQRT,
    FDIV_ROUND,
    FDIV_RSQRT,
    FDIV_SQRT,
    FMIN_HELPERS,
    FROUND_DOUBLE,
    FROUND_FLOAT,
    FROUND_HELPERS,
    FSPLIT_HELPERS,
    HYPOT_HELPERS,
    LROUND_HELPERS,
    REMQUO_HELPERS,
    SCALE_HELPERS,
    fround_needs_double,
    fround_needs_float,
    fsplit_arity,
    fsplit_blocks,
    hypot_arity,
    hypot_blocks,
    lround_blocks,
    remquo_blocks,
    scale_arity,
    scale_blocks,
    FMIN_LESS,
    FMIN_MINMAX,
    FMIN_NEXT,
    DMIN_LESS,
    DMIN_MINMAX,
    DMIN_NEXT,
    DARITH_ADD,
    DARITH_FMA,
    DARITH_HELPERS,
    DARITH_MUL,
    DARITH_PACK,
    DARITH_SUB,
    FARITH_ADD,
    FARITH_FMA,
    FARITH_HELPERS,
    FARITH_MUL,
    FARITH_PACK,
    HALF_FROM_DOUBLE,
    HALF_FROM_FLOAT,
    HALF_FROM_INT,
    HALF_HELPERS,
    HALF_TO_FLOAT,
    HALF_TO_INT,
    HARITH_ADD,
    HARITH_FMA,
    HARITH_HELPERS,
    HARITH_MUL,
    HARITH_SAT,
    H2_ADD,
    H2_FMA,
    H2_HELPERS,
    H2_MUL,
    H2_SAT,
    HPRED_CMP,
    HPRED_HELPERS,
    HPRED_MINMAX,
    HPRED_NAN,
    HPRED_SIGN,
    Dialect,
    ddiv_needs_div,
    ddiv_needs_sqrt,
    fdiv_needs_div,
    fdiv_needs_rsqrt,
    fdiv_needs_sqrt,
    fmin_needs_dless,
    fmin_needs_dminmax,
    fmin_needs_dnext,
    fmin_needs_fless,
    fmin_needs_fminmax,
    fmin_needs_fnext,
    darith_needs_add,
    darith_needs_fma,
    darith_needs_mul,
    darith_needs_sub,
    farith_needs_add,
    farith_needs_fma,
    farith_needs_mul,
    half_needs_double,
    half_needs_from_float,
    half_needs_int,
    half_needs_to_float,
    half_needs_to_int,
    harith_needs_add,
    harith_needs_fma,
    harith_needs_mul,
    harith_needs_sat,
    h2_arity,
    h2_needs_add,
    h2_needs_fma,
    h2_needs_from_float,
    h2_needs_mul,
    h2_needs_sat,
    h2_needs_to_float,
    hpred_arity,
    hpred_needs_cmp,
    hpred_needs_minmax,
    hpred_needs_sign,
    PTX_BFE_S32,
    PTX_BFE_U32,
    PTX_BFI,
    PTX_LOP3,
    PTX_PRMT,
    round_needs_bits,
    round_needs_d2f,
    round_needs_i2d_s64,
    round_needs_i2d_u64,
    round_needs_i32,
    round_needs_i64,
    runtime_name,
)
from cuda_retarget.parse import ParseError, Tok, parse_file

_FULL_MASKS = {"0xffffffff", "0xffffffffu", "0xffffffffl", "0xffffffffll", "0xffffffffull"}
_PTX_LOP3 = re.compile(r"^lop3\.b32%0,%1,%2,%3,(0[xX][0-9a-fA-F]+|[0-9]+);$")
_PTX_BFE_U32 = re.compile(r"^bfe\.u32%0,%1,%2,%3;$")
_PTX_BFE_S32 = re.compile(r"^bfe\.s32%0,%1,%2,%3;$")
_PTX_BFI = re.compile(r"^bfi\.b32%0,%1,%2,%3,%4;$")
_PTX_PRMT = re.compile(r"^prmt\.b32%0,%1,%2,%3;$")
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
        asm = _asm_edit(text, tokens, index, dialect)
        if asm is not None:
            edits.append(asm[0])
            index = asm[1]
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


def _asm_constraint(spelling: str) -> str:
    if len(spelling) >= 2 and spelling[0] == '"' and spelling[-1] == '"':
        return spelling[1:-1].strip()
    return spelling.strip()


def _ptx_form(body: str) -> tuple[str, int, str] | None:
    compact = "".join(body.split())
    lop3 = _PTX_LOP3.fullmatch(compact)
    if lop3 is not None:
        immediate = lop3.group(1)
        if immediate.lower().startswith("0x"):
            value = int(immediate, 16)
        else:
            value = int(immediate, 10)
        if value > 255:
            raise TranslateError(f"lop3.b32 immediate {immediate} is wider than 8 bits")
        return "cts_lop3", 3, immediate
    if _PTX_BFE_U32.fullmatch(compact):
        return "cts_bfe_u32", 3, ""
    if _PTX_BFE_S32.fullmatch(compact):
        return "cts_bfe_s32", 3, ""
    if _PTX_BFI.fullmatch(compact):
        return "cts_bfi", 4, ""
    if _PTX_PRMT.fullmatch(compact):
        return "cts_prmt", 3, ""
    return None


def _asm_operand(
    tokens: list[Tok], cursor: int
) -> tuple[str, int, int, int] | None:
    if cursor + 1 >= len(tokens) or tokens[cursor].kind != "LITERAL":
        return None
    if tokens[cursor + 1].spelling != "(":
        return None
    depth = 0
    expr_start: int | None = None
    for index in range(cursor + 1, len(tokens)):
        spelling = tokens[index].spelling
        if spelling == "(":
            depth += 1
            if depth == 1:
                expr_start = tokens[index + 1].start if index + 1 < len(tokens) else None
            continue
        if spelling == ")":
            depth -= 1
            if depth == 0:
                if expr_start is None or index == cursor + 2:
                    return None
                return _asm_constraint(tokens[cursor].spelling), expr_start, tokens[index - 1].end, index + 1
            continue
    return None


def _operand_text(
    text: str, tokens: list[Tok], start: int, end: int, dialect: Dialect
) -> str:
    edits: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.start < start or token.end > end or token.kind != "IDENTIFIER":
            continue
        if (
            dialect.shuffle_width_32
            and token.spelling == "warpSize"
            and not _is_define_name(tokens, index)
        ):
            edits.append((token.start, token.end, "32"))
            continue
        renamed = runtime_name(token.spelling, dialect)
        if renamed is not None:
            edits.append((token.start, token.end, renamed))
    data = bytearray(text.encode()[start:end])
    for edit_start, edit_end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        data[edit_start - start : edit_end - start] = replacement.encode()
    return data.decode()


def _asm_edit(
    text: str, tokens: list[Tok], index: int, dialect: Dialect
) -> tuple[tuple[int, int, str], int] | None:
    if not dialect.lower_ptx or tokens[index].spelling != "asm":
        return None
    cursor = index + 1
    if cursor < len(tokens) and tokens[cursor].spelling == "volatile":
        cursor += 1
    if cursor >= len(tokens) or tokens[cursor].spelling != "(":
        return None
    cursor += 1
    if cursor >= len(tokens) or tokens[cursor].kind != "LITERAL":
        return None
    literal = tokens[cursor].spelling
    if len(literal) < 2 or literal[0] != '"' or literal[-1] != '"':
        return None
    cursor += 1
    if cursor < len(tokens) and tokens[cursor].kind == "LITERAL":
        return None
    try:
        form = _ptx_form(literal[1:-1])
    except TranslateError:
        raise
    if form is None:
        return None
    helper, inputs, immediate = form
    if cursor >= len(tokens) or tokens[cursor].spelling != ":":
        raise TranslateError(f"inline {helper} is not a single =r output")
    cursor += 1
    output = _asm_operand(tokens, cursor)
    if output is None:
        raise TranslateError(f"inline {helper} is not a single =r output")
    constraint, expr_start, expr_end, cursor = output
    if constraint != "=r":
        raise TranslateError(f"inline {helper} output constraint {constraint} is not =r")
    if cursor < len(tokens) and tokens[cursor].spelling == ",":
        raise TranslateError(f"inline {helper} has more than one output")
    if cursor >= len(tokens) or tokens[cursor].spelling != ":":
        raise TranslateError(f"inline {helper} inputs are not r operands")
    cursor += 1
    arguments: list[str] = []
    for number in range(inputs):
        if number:
            if cursor >= len(tokens) or tokens[cursor].spelling != ",":
                raise TranslateError(f"inline {helper} needs {inputs} r operands")
            cursor += 1
        operand = _asm_operand(tokens, cursor)
        if operand is None:
            raise TranslateError(f"inline {helper} needs {inputs} r operands")
        in_constraint, in_start, in_end, cursor = operand
        if in_constraint != "r":
            raise TranslateError(f"inline {helper} input constraint {in_constraint} is not r")
        arguments.append(_operand_text(text, tokens, in_start, in_end, dialect))
    if cursor < len(tokens) and tokens[cursor].spelling in {",", ":"}:
        raise TranslateError(f"inline {helper} is not the single-instruction form")
    if cursor >= len(tokens) or tokens[cursor].spelling != ")":
        raise TranslateError(f"inline {helper} is not the single-instruction form")
    if immediate:
        arguments.append(immediate)
    destination = _operand_text(text, tokens, expr_start, expr_end, dialect)
    replacement = f"{destination} = {helper}({', '.join(arguments)})"
    return (tokens[index].start, tokens[cursor].end, replacement), cursor + 1


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
    if (
        dialect.lower_hadd
        and token.spelling in HADD_HELPERS
        and not (token.spelling == "__hadd" and _is_half_result(token.call_type))
    ):
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
    if dialect.lower_half and token.spelling in HALF_HELPERS:
        parsed = _arguments(tokens, index + 1)
        if parsed is None or len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = HALF_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_half and token.spelling in HARITH_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 3 if token.spelling.startswith("__hfma") else 2
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = HARITH_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_half and token.spelling in H2_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = h2_arity(token.spelling)
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = H2_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_half and token.spelling in HPRED_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = hpred_arity(token.spelling)
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = HPRED_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_fdiv and token.spelling in FDIV_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 2 if token.spelling.startswith("__fdiv_") else 1
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = FDIV_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_fdiv and token.spelling in DDIV_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 2 if token.spelling.startswith("__ddiv_") else 1
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = DDIV_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_fmin and token.spelling in FMIN_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != 2:
            raise TranslateError(f"{token.spelling} must have two arguments")
        replacement, _helper = FMIN_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_fround and token.spelling in FROUND_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = FROUND_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_fsplit and token.spelling in FSPLIT_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != fsplit_arity(token.spelling):
            raise TranslateError(f"{token.spelling} must have {fsplit_arity(token.spelling)} arguments")
        replacement, _helper = FSPLIT_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_lround and token.spelling in LROUND_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != 1:
            raise TranslateError(f"{token.spelling} must have one argument")
        replacement, _helper = LROUND_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_hypot and token.spelling in HYPOT_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != hypot_arity(token.spelling):
            raise TranslateError(f"{token.spelling} must have {hypot_arity(token.spelling)} arguments")
        replacement, _helper = HYPOT_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_remquo and token.spelling in REMQUO_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != 3:
            raise TranslateError(f"{token.spelling} must have three arguments")
        replacement, _helper = REMQUO_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_scale and token.spelling in SCALE_HELPERS:
        if _is_define_name(tokens, index) or _is_declared_here(tokens, index):
            return None
        parsed = _arguments(tokens, index + 1)
        if parsed is None:
            return None
        if len(parsed[0]) != scale_arity(token.spelling):
            raise TranslateError(f"{token.spelling} must have {scale_arity(token.spelling)} arguments")
        replacement, _helper = SCALE_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_darith and token.spelling in DARITH_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 3 if token.spelling.startswith("__fma_") else 2
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = DARITH_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    if dialect.lower_farith and token.spelling in FARITH_HELPERS:
        parsed = _arguments(tokens, index + 1)
        arity = 3 if token.spelling.startswith("__fmaf_") else 2
        if parsed is None or len(parsed[0]) != arity:
            raise TranslateError(f"{token.spelling} must have {arity} arguments")
        replacement, _helper = FARITH_HELPERS[token.spelling]
        return (token.start, token.end, replacement)
    return None


def _is_half_result(spelling: str) -> bool:
    text = spelling.strip()
    if text.startswith("const "):
        text = text[len("const ") :].strip()
    return text == "__half"


_TYPE_BEFORE = {
    "float",
    "double",
    "int",
    "void",
    "auto",
    "char",
    "short",
    "long",
    "unsigned",
    "signed",
    "bool",
    "const",
    "volatile",
    "static",
    "__device__",
    "__host__",
    "__global__",
    "__forceinline__",
    "__inline__",
    "*",
    "&",
    "&&",
}


def _is_define_name(tokens: list[Tok], index: int) -> bool:
    return index >= 2 and tokens[index - 2].spelling == "#" and tokens[index - 1].spelling == "define"


def _is_declared_here(tokens: list[Tok], index: int) -> bool:
    return index > 0 and tokens[index - 1].spelling in _TYPE_BEFORE


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
        if round_needs_i2d_u64(names):
            blocks.append(I2D_U64.strip("\n"))
        if round_needs_i2d_s64(names):
            blocks.append(I2D_S64.strip("\n"))
        for _name, helper in used:
            blocks.append(helper.strip("\n"))
    if dialect.lower_saturate:
        for name, helper in SATURATE_HELPERS.values():
            if f"{name}(" in rewritten:
                blocks.append(helper.strip("\n"))
    if dialect.lower_half:
        used = [(name, helper) for name, helper in HALF_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            if half_needs_from_float(names):
                blocks.append(HALF_FROM_FLOAT.strip("\n"))
            if half_needs_int(names):
                blocks.append(HALF_FROM_INT.strip("\n"))
            if half_needs_double(names):
                blocks.append(HALF_FROM_DOUBLE.strip("\n"))
            if half_needs_to_float(names):
                blocks.append(HALF_TO_FLOAT.strip("\n"))
            if half_needs_to_int(names):
                blocks.append(HALF_TO_INT.strip("\n"))
            seen: set[str] = set()
            for name, helper in used:
                if name in seen:
                    continue
                seen.add(name)
                blocks.append(helper.strip("\n"))
        harith = [(name, helper) for name, helper in HARITH_HELPERS.values() if f"{name}(" in rewritten]
        harith_names = {name for name, _helper in harith}
        if harith_names:
            if "cts_hpack(" not in "\n".join(blocks):
                blocks.append(HALF_FROM_FLOAT.strip("\n"))
            if harith_needs_add(harith_names):
                blocks.append(HARITH_ADD.strip("\n"))
            if harith_needs_mul(harith_names):
                blocks.append(HARITH_MUL.strip("\n"))
            if harith_needs_fma(harith_names):
                blocks.append(HARITH_FMA.strip("\n"))
            if harith_needs_sat(harith_names):
                blocks.append(HARITH_SAT.strip("\n"))
            seen_harith: set[str] = set()
            for name, helper in harith:
                if name in seen_harith:
                    continue
                seen_harith.add(name)
                blocks.append(helper.strip("\n"))
        h2 = [(name, helper) for name, helper in H2_HELPERS.values() if f"{name}(" in rewritten]
        h2_names = {name for name, _helper in h2}
        if h2_names:
            joined = "\n".join(blocks)
            needs_pack = (
                h2_needs_from_float(h2_names)
                or h2_needs_add(h2_names)
                or h2_needs_mul(h2_names)
                or h2_needs_fma(h2_names)
            )
            if needs_pack and "cts_hpack(" not in joined:
                blocks.append(HALF_FROM_FLOAT.strip("\n"))
                joined = "\n".join(blocks)
            if h2_needs_to_float(h2_names) and "cts_h2f(" not in joined:
                blocks.append(HALF_TO_FLOAT.strip("\n"))
                joined = "\n".join(blocks)
            if (h2_needs_add(h2_names) or h2_needs_fma(h2_names)) and (
                "static __device__ unsigned short cts_f16add(" not in joined
            ):
                blocks.append(HARITH_ADD.strip("\n"))
                joined = "\n".join(blocks)
            if h2_needs_mul(h2_names) and "static __device__ unsigned short cts_f16mul(" not in joined:
                blocks.append(HARITH_MUL.strip("\n"))
                joined = "\n".join(blocks)
            if h2_needs_fma(h2_names) and "static __device__ unsigned short cts_f16fma(" not in joined:
                blocks.append(HARITH_FMA.strip("\n"))
                joined = "\n".join(blocks)
            if h2_needs_sat(h2_names) and "static __device__ unsigned short cts_f16sat(" not in joined:
                blocks.append(HARITH_SAT.strip("\n"))
            if h2_needs_add(h2_names):
                blocks.append(H2_ADD.strip("\n"))
            if h2_needs_mul(h2_names):
                blocks.append(H2_MUL.strip("\n"))
            if h2_needs_fma(h2_names):
                blocks.append(H2_FMA.strip("\n"))
            if h2_needs_sat(h2_names):
                blocks.append(H2_SAT.strip("\n"))
            seen_h2: set[str] = set()
            for name, helper in h2:
                if name in seen_h2:
                    continue
                seen_h2.add(name)
                blocks.append(helper.strip("\n"))
        pred = [(name, helper) for name, helper in HPRED_HELPERS.values() if f"{name}(" in rewritten]
        pred_names = {name for name, _helper in pred}
        if pred_names:
            joined = "\n".join(blocks)
            blocks.append(HPRED_NAN.strip("\n"))
            if hpred_needs_sign(pred_names):
                blocks.append(HPRED_SIGN.strip("\n"))
            if (hpred_needs_cmp(pred_names) or hpred_needs_minmax(pred_names)) and "cts_h2f(" not in joined:
                blocks.append(HALF_TO_FLOAT.strip("\n"))
            if hpred_needs_cmp(pred_names):
                blocks.append(HPRED_CMP.strip("\n"))
            if hpred_needs_minmax(pred_names):
                blocks.append(HPRED_MINMAX.strip("\n"))
            seen_pred: set[str] = set()
            for name, helper in pred:
                if name in seen_pred:
                    continue
                seen_pred.add(name)
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
    if dialect.lower_darith:
        used = [(name, helper) for name, helper in DARITH_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.append(DARITH_PACK.strip("\n"))
            if darith_needs_add(names):
                blocks.append(DARITH_ADD.strip("\n"))
            if darith_needs_sub(names):
                blocks.append(DARITH_SUB.strip("\n"))
            if darith_needs_mul(names):
                blocks.append(DARITH_MUL.strip("\n"))
            if darith_needs_fma(names):
                blocks.append(DARITH_FMA.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_fdiv:
        used = [(name, helper) for name, helper in FDIV_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            joined = "\n".join(blocks)
            if "static __device__ unsigned cts_fpack(" not in joined:
                blocks.append(FARITH_PACK.strip("\n"))
            blocks.append(FDIV_ROUND.strip("\n"))
            if fdiv_needs_div(names):
                blocks.append(FDIV_DIV.strip("\n"))
            if fdiv_needs_sqrt(names) or fdiv_needs_rsqrt(names):
                blocks.append(FDIV_ISQRT.strip("\n"))
            if fdiv_needs_sqrt(names):
                blocks.append(FDIV_SQRT.strip("\n"))
            if fdiv_needs_rsqrt(names):
                blocks.append(FDIV_RSQRT.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_fdiv:
        used = [(name, helper) for name, helper in DDIV_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            joined = "\n".join(blocks)
            if "static __device__ unsigned long long cts_dpack(" not in joined:
                blocks.append(DARITH_PACK.strip("\n"))
            if ddiv_needs_div(names):
                blocks.append(DDIV_DIV.strip("\n"))
            if ddiv_needs_sqrt(names):
                if "cts_isqrt(" not in joined:
                    blocks.append(FDIV_ISQRT.strip("\n"))
                blocks.append(DDIV_SQRT.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_fmin:
        used = [(name, helper) for name, helper in FMIN_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            if fmin_needs_fless(names):
                blocks.append(FMIN_LESS.strip("\n"))
            if fmin_needs_fminmax(names):
                blocks.append(FMIN_MINMAX.strip("\n"))
            if fmin_needs_fnext(names):
                blocks.append(FMIN_NEXT.strip("\n"))
            if fmin_needs_dless(names):
                blocks.append(DMIN_LESS.strip("\n"))
            if fmin_needs_dminmax(names):
                blocks.append(DMIN_MINMAX.strip("\n"))
            if fmin_needs_dnext(names):
                blocks.append(DMIN_NEXT.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_fround:
        used = [(name, helper) for name, helper in FROUND_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            if fround_needs_float(names):
                blocks.append(FROUND_FLOAT.strip("\n"))
            if fround_needs_double(names):
                blocks.append(FROUND_DOUBLE.strip("\n"))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_fsplit:
        used = [(name, helper) for name, helper in FSPLIT_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.extend(fsplit_blocks(names, "\n".join(blocks)))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_lround:
        used = [(name, helper) for name, helper in LROUND_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.extend(lround_blocks(names, "\n".join(blocks)))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_hypot:
        used = [(name, helper) for name, helper in HYPOT_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.extend(hypot_blocks(names, "\n".join(blocks)))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_remquo:
        used = [(name, helper) for name, helper in REMQUO_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.extend(remquo_blocks(names, "\n".join(blocks)))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_scale:
        used = [(name, helper) for name, helper in SCALE_HELPERS.values() if f"{name}(" in rewritten]
        names = {name for name, _helper in used}
        if names:
            blocks.extend(scale_blocks(names, "\n".join(blocks)))
            for _name, helper in used:
                blocks.append(helper.strip("\n"))
    if dialect.lower_ptx:
        if "cts_lop3(" in rewritten:
            blocks.append(PTX_LOP3.strip("\n"))
        if "cts_bfe_u32(" in rewritten:
            blocks.append(PTX_BFE_U32.strip("\n"))
        if "cts_bfe_s32(" in rewritten:
            blocks.append(PTX_BFE_S32.strip("\n"))
        if "cts_bfi(" in rewritten:
            blocks.append(PTX_BFI.strip("\n"))
        if "cts_prmt(" in rewritten:
            blocks.append(PTX_PRMT.strip("\n"))
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
