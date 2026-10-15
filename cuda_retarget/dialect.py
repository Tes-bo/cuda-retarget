# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Target dialects. Runtime names change. A few CUDA assumptions do not survive a rename."""

from __future__ import annotations

import re
from dataclasses import dataclass

# cudaMalloc, cudaError_t, cudaSuccess. Not cuda_runtime, which is a header name.
_RUNTIME_NAME = re.compile(r"cuda(?=[A-Z])")

# Intrinsic name, replacement name, device helper. The helper keeps the IEEE bits.
BITCAST_HELPERS = {
    "__double_as_longlong": (
        "cts_double_bits",
        """
static __device__ unsigned long long cts_double_bits(double value) {
    union {
        double as_double;
        unsigned long long as_bits;
    } converted;
    converted.as_double = value;
    return converted.as_bits;
}
""",
    ),
    "__longlong_as_double": (
        "cts_longlong_bits",
        """
static __device__ double cts_longlong_bits(unsigned long long bits) {
    union {
        unsigned long long as_bits;
        double as_double;
    } converted;
    converted.as_bits = bits;
    return converted.as_double;
}
""",
    ),
    "__float_as_int": (
        "cts_float_bits",
        """
static __device__ int cts_float_bits(float value) {
    union {
        float as_float;
        int as_bits;
    } converted;
    converted.as_float = value;
    return converted.as_bits;
}
""",
    ),
    "__int_as_float": (
        "cts_int_bits",
        """
static __device__ float cts_int_bits(int bits) {
    union {
        int as_bits;
        float as_float;
    } converted;
    converted.as_bits = bits;
    return converted.as_float;
}
""",
    ),
    "__float_as_uint": (
        "cts_float_ubits",
        """
static __device__ unsigned int cts_float_ubits(float value) {
    union {
        float as_float;
        unsigned int as_bits;
    } converted;
    converted.as_float = value;
    return converted.as_bits;
}
""",
    ),
    "__uint_as_float": (
        "cts_uint_bits",
        """
static __device__ float cts_uint_bits(unsigned int bits) {
    union {
        unsigned int as_bits;
        float as_float;
    } converted;
    converted.as_bits = bits;
    return converted.as_float;
}
""",
    ),
}


# Concatenate hi:lo, then shift. Wrap uses shift & 31. Clamp uses min(shift, 32).
FUNNEL_HELPERS = {
    "__funnelshift_l": (
        "cts_funnelshift_l",
        """
static __device__ unsigned cts_funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {
    unsigned amt = shift & 31u;
    unsigned long long value = (static_cast<unsigned long long>(hi) << 32) | lo;
    return static_cast<unsigned>((value << amt) >> 32);
}
""",
    ),
    "__funnelshift_lc": (
        "cts_funnelshift_lc",
        """
static __device__ unsigned cts_funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {
    unsigned amt = shift > 32u ? 32u : shift;
    unsigned long long value = (static_cast<unsigned long long>(hi) << 32) | lo;
    return static_cast<unsigned>((value << amt) >> 32);
}
""",
    ),
    "__funnelshift_r": (
        "cts_funnelshift_r",
        """
static __device__ unsigned cts_funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {
    unsigned amt = shift & 31u;
    unsigned long long value = (static_cast<unsigned long long>(hi) << 32) | lo;
    return static_cast<unsigned>(value >> amt);
}
""",
    ),
    "__funnelshift_rc": (
        "cts_funnelshift_rc",
        """
static __device__ unsigned cts_funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {
    unsigned amt = shift > 32u ? 32u : shift;
    unsigned long long value = (static_cast<unsigned long long>(hi) << 32) | lo;
    return static_cast<unsigned>(value >> amt);
}
""",
    ),
}


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
    `lower_bitcasts` replaces the CUDA float and double bitcast intrinsics
    with union helpers. The IEEE bit pattern of the value does not change.
    `match_width_32` lowers a full-mask `__match_any_sync` or
    `__match_all_sync` to a 32-lane helper. CUDA's result is a mask of one
    warp. A 64-lane wavefront would include the paired lane from the next
    warp.
    `vote_width_32` lowers a full-mask `__ballot_sync`, `__any_sync`,
    `__all_sync`, or `__uni_sync` the same way. A predicate that is uniform
    inside one warp of 32 can be mixed across a 64-lane wavefront.
    `syncwarp_mask_32` rewrites a no-argument `__syncwarp()` to
    `__syncwarp(0xffffffff)`. CUDA's default mask names 32 lanes. A header
    default on a 64-lane wavefront does not.
    `lower_funnelshifts` replaces `__funnelshift_l`, `__funnelshift_lc`,
    `__funnelshift_r`, and `__funnelshift_rc` with 64-bit shift helpers.
    Wrap (`l`, `r`) uses `shift & 31`. Clamp (`lc`, `rc`) uses `min(shift, 32)`.
    """

    name: str
    prefix: str | None
    headers: dict[str, str]
    shuffle_width_32: bool
    lower_bitcasts: bool
    match_width_32: bool
    vote_width_32: bool
    syncwarp_mask_32: bool
    lower_funnelshifts: bool


DIALECTS = {
    "hip": Dialect(
        name="hip",
        prefix="hip",
        headers={"cuda_runtime.h": "hip/hip_runtime.h"},
        shuffle_width_32=True,
        lower_bitcasts=True,
        match_width_32=True,
        vote_width_32=True,
        syncwarp_mask_32=True,
        lower_funnelshifts=True,
    ),
    "musa": Dialect(
        name="musa",
        prefix="musa",
        headers={"cuda_runtime.h": "musa_runtime.h"},
        shuffle_width_32=False,
        lower_bitcasts=True,
        match_width_32=False,
        vote_width_32=False,
        syncwarp_mask_32=False,
        lower_funnelshifts=True,
    ),
    "maca": Dialect(
        name="maca",
        prefix=None,
        headers={},
        shuffle_width_32=False,
        lower_bitcasts=False,
        match_width_32=False,
        vote_width_32=False,
        syncwarp_mask_32=False,
        lower_funnelshifts=False,
    ),
}


MATCH_ANY_HELPER = """
template <class T>
static __device__ int cts_same_bits(T left, T right) {
    return left == right;
}

static __device__ int cts_same_bits(float left, float right) {
    union {
        float as_float;
        unsigned as_bits;
    } a, b;
    a.as_float = left;
    b.as_float = right;
    return a.as_bits == b.as_bits;
}

static __device__ int cts_same_bits(double left, double right) {
    union {
        double as_double;
        unsigned long long as_bits;
    } a, b;
    a.as_double = left;
    b.as_double = right;
    return a.as_bits == b.as_bits;
}

template <class T>
static __device__ unsigned cts_match_any_sync(unsigned mask, T value) {
    unsigned bits = 0;
#pragma unroll
    for (int src = 0; src < 32; ++src) {
        T other = __shfl_sync(mask, value, src, 32);
        if (cts_same_bits(other, value))
            bits |= 1u << src;
    }
    return bits;
}
"""

MATCH_ALL_HELPER = """
template <class T>
static __device__ unsigned cts_match_all_sync(unsigned mask, T value, int* pred) {
    unsigned same = cts_match_any_sync(mask, value) == mask;
    *pred = static_cast<int>(same);
    return same ? mask : 0u;
}
"""


VOTE_HELPERS = {
    "cts_ballot_sync": """
static __device__ unsigned cts_ballot_sync(unsigned mask, int predicate) {
    unsigned bits = 0;
#pragma unroll
    for (int src = 0; src < 32; ++src) {
        int other = __shfl_sync(mask, predicate, src, 32);
        if (other)
            bits |= 1u << src;
    }
    return bits;
}
""",
    "cts_any_sync": """
static __device__ int cts_any_sync(unsigned mask, int predicate) {
    return cts_ballot_sync(mask, predicate) != 0;
}
""",
    "cts_all_sync": """
static __device__ int cts_all_sync(unsigned mask, int predicate) {
    return cts_ballot_sync(mask, predicate) == mask;
}
""",
    "cts_uni_sync": """
static __device__ int cts_uni_sync(unsigned mask, int predicate) {
    unsigned bits = cts_ballot_sync(mask, predicate);
    return bits == 0u || bits == mask;
}
""",
}


def runtime_name(spelling: str, dialect: Dialect) -> str | None:
    if dialect.prefix is None or _RUNTIME_NAME.match(spelling) is None:
        return None
    return dialect.prefix + spelling[4:]
