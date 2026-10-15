# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Target dialects. Runtime names change. A few CUDA assumptions do not survive a rename."""

from __future__ import annotations

import re
from dataclasses import dataclass

# cudaMalloc, cudaError_t, cudaSuccess. Not cuda_runtime, which is a header name.
_RUNTIME_NAME = re.compile(r"cuda(?=[A-Z])")

DOUBLE_BITS_HELPER = """
static __device__ unsigned long long cts_double_bits(double value) {
    union {
        double as_double;
        unsigned long long as_bits;
    } converted;
    converted.as_double = value;
    return converted.as_bits;
}
"""


@dataclass(frozen=True)
class Dialect:
    """One emission target.

    `prefix` replaces the `cuda` prefix on runtime identifiers. None means the
    vendor compiler accepts CUDA spellings, so names stay.
    `shuffle_width_32` keeps CUDA's subgroup of 32 lanes. A 3-argument
    full-mask `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, or
    `__shfl_down_sync` gains an explicit width of 32, and a `warpSize`
    use becomes the literal 32. NVIDIA's default width is 32. AMD's
    default wavefront is 64, so leaving either one alone is a different
    program. A `#define` name is left in place.
    `lower_double_bits` replaces `__double_as_longlong` with a union helper.
    The IEEE bit pattern of the value does not change.
    """

    name: str
    prefix: str | None
    headers: dict[str, str]
    shuffle_width_32: bool
    lower_double_bits: bool


DIALECTS = {
    "hip": Dialect(
        name="hip",
        prefix="hip",
        headers={"cuda_runtime.h": "hip/hip_runtime.h"},
        shuffle_width_32=True,
        lower_double_bits=True,
    ),
    "musa": Dialect(
        name="musa",
        prefix="musa",
        headers={"cuda_runtime.h": "musa_runtime.h"},
        shuffle_width_32=False,
        lower_double_bits=True,
    ),
    "maca": Dialect(
        name="maca",
        prefix=None,
        headers={},
        shuffle_width_32=False,
        lower_double_bits=False,
    ),
}


def runtime_name(spelling: str, dialect: Dialect) -> str | None:
    if dialect.prefix is None or _RUNTIME_NAME.match(spelling) is None:
        return None
    return dialect.prefix + spelling[4:]
