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


# High and low 32-bit words of an IEEE double. Same bits as the 64-bit bitcast.
DOUBLE_WORD_HELPERS = {
    "__double2hiint": (
        "cts_double_hiint",
        """
static __device__ int cts_double_hiint(double value) {
    union {
        double as_double;
        unsigned long long as_bits;
    } converted;
    union {
        unsigned as_unsigned;
        int as_int;
    } high;
    converted.as_double = value;
    high.as_unsigned = static_cast<unsigned>(converted.as_bits >> 32);
    return high.as_int;
}
""",
    ),
    "__double2loint": (
        "cts_double_loint",
        """
static __device__ int cts_double_loint(double value) {
    union {
        double as_double;
        unsigned long long as_bits;
    } converted;
    union {
        unsigned as_unsigned;
        int as_int;
    } low;
    converted.as_double = value;
    low.as_unsigned = static_cast<unsigned>(converted.as_bits);
    return low.as_int;
}
""",
    ),
    "__hiloint2double": (
        "cts_hiloint_double",
        """
static __device__ double cts_hiloint_double(int hi, int lo) {
    union {
        unsigned as_unsigned;
        int as_int;
    } high, low;
    union {
        unsigned long long as_bits;
        double as_double;
    } converted;
    high.as_int = hi;
    low.as_int = lo;
    converted.as_bits = (static_cast<unsigned long long>(high.as_unsigned) << 32) | low.as_unsigned;
    return converted.as_double;
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
    with union helpers, and splits or joins a double's high and low 32-bit
    words (`__double2hiint`, `__double2loint`, `__hiloint2double`). The
    IEEE bit pattern of the value does not change.
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
    `lower_dp4a` replaces `__dp4a` with signed and unsigned four-byte
    dot-product helpers. A 0x80 byte is -128 on the signed overload and
    128 on the unsigned overload.
    `lower_packed_arith` replaces `__vadd2`, `__vaddss2`, `__vaddus2`,
    `__vsub2`, `__vsubss2`, `__vsubus2`, and the 4-byte forms. Plain
    calls wrap each lane. `ss` saturates as signed. `us` saturates as
    unsigned. A 32-bit add is a different program.
    `lower_packed_minmax` replaces `__vmins2`, `__vminu2`, `__vmaxs2`,
    `__vmaxu2`, and the 4-byte forms. `s` compares each lane as signed.
    `u` compares it as unsigned. `0x8000` is -32768 on one and 32768
    on the other.
    `lower_packed_abs` replaces per-lane absolute value, negation, and
    absolute difference. Plain `__vabs2` / `__vabs4` wrap, so the minimum
    lane stays the minimum bit pattern. The `ss` absolute value saturates
    that lane to the maximum. `__vnegss2` does not: a `0x8000` lane stays
    `0x8000`, the same bits as `__vneg2`. `__vnegss4` saturates `0x80` to
    `0x7F`.     Signed and unsigned absolute differences are different lanes.
    `lower_byte_perm` replaces `__byte_perm` with a helper that
    concatenates `y:x` and picks four bytes. Each selector is three bits
    from the low 16 bits of the third argument. The high bit of each
    nibble, and every bit above bit 15, is ignored.
    `lower_hadd` replaces `__hadd`, `__rhadd`, `__uhadd`, and `__urhadd`.
    The plain forms are `(x + y) >> 1`. The rounded forms add one before
    the shift.     The sum is computed without a 32-bit overflow, so the
    average of two minimum signed values stays negative and the average
    of `0xFFFFFFFF` and `1` is `2^31`.
    `lower_mulhi` replaces `__mulhi`, `__umulhi`, `__mul64hi`, and
    `__umul64hi` with the high half of the full product. The 32-bit
    forms widen to 64 bits. The 64-bit forms build a 128-bit product
    from 32-bit halves. A negative product sign-extends, so the high
    half of a small negative product is all ones. `INT_MIN * INT_MIN`
    keeps the high half `2^30` instead of wrapping a 32-bit multiply.
    `lower_sad` replaces `__sad` and `__usad`. Each result is the
    absolute difference plus the third argument. The signed form widens
    before the subtraction, so the absolute value of `INT_MIN` is
    `2^31` and `|INT_MIN - 1|` is `2^31 + 1`. The unsigned form treats
    the same bits as a magnitude. The sum wraps modulo `2^32`.
    `lower_packed_sad` replaces `__vsads2`, `__vsadu2`, `__vsads4`, and
    `__vsadu4` with the sum of the per-lane absolute differences. A
    signed `0xFFFF` lane is 1 and an unsigned one is 65535. A `0x8000`
    lane contributes 32768, so two of them sum to 65536. The result is
    a scalar sum, not a packed word.
    `lower_packed_cmp` replaces the `__vcmp*` lane comparisons with a
    mask. A true halfword lane is filled with `0xFFFF` and a true byte
    lane with `0xFF`. Ordering comparisons treat `0x8000` as signed or
    unsigned. Equality compares the lane bits.
    `lower_packed_set` replaces the `__vset*` lane predicates. A true
    halfword writes `0x0001` and a true byte writes `0x01`. That is not
    the all-ones mask from `__vcmp*`, and lanes that fail stay zero
    instead of clearing the whole result.
    `lower_packed_avg` replaces `__vavgs2`, `__vavgs4`, `__vavgu2`,
    `__vavgu4`, `__vhaddu2`, and `__vhaddu4`. The signed average rounds
    a half away from zero, so 1 and 2 become 2 while -2 and -3 become
    -3. Two minimum lanes stay the minimum. The unsigned rounded form
    does the same for magnitudes. The plain unsigned average truncates,
    so `0xFFFF` and 1 become `0x8000` rather than 0.
    `lower_bitops` replaces `__clz`, `__ffs`, `__popc`, `__brev`, and
    the 64-bit forms. `__clz(0)` is 32 and `__clzll(0)` is 64. `__ffs(0)`
    is 0, and a set bit is counted from 1. `__brev` reverses the bits.
    `lower_mul24` replaces `__mul24` and `__umul24`. The high 8 bits of
    each input are ignored. Bit 23 is the sign of the signed form. The
    low 32 bits of the product are kept, so the square of the minimum
    24-bit value is 0.
    `lower_round` replaces the rounding conversions between float and
    32-bit or 64-bit integers, and the double conversions to float and
    to integers. `__float2int_rn(2.5)` is 2 and `__float2int_ru(2.5)` is
    3. A value past the integer range saturates. A NaN becomes 0 for
    `__float2int_*` and the high bit for `__float2ll_*`, `__float2ull_*`,
    and every `__double2int_*` mode. `__double2float_rn` of the halfway
    value `1+2^-24` stays on 1, and round-up takes the next float. A
    double past the finite float maximum stays at that maximum when
    rounding toward zero. A double NaN becomes a quiet float NaN with
    the same sign. Every 32-bit integer is exact in double, so the
    int-to-float helpers round that double. A long long past 2^53 is not
    exact in double, so those helpers round from the integer bits.
    `__int2double_rn` and `__uint2double_rn` are exact, including `INT_MIN`.
    `__ll2double_rn` of `2^53+1` stays on `2^53`, and round-up takes the next
    double. The same value negated, rounded toward -inf, is one double below
    the nearest-even result. `__ull2double_rz` of the maximum unsigned value
    is `2^64-2048`; round-up reaches `2^64`.
    `lower_saturate` replaces `__saturatef`. Values outside [0, 1] clamp,
    and a NaN or a negative zero becomes +0.
    `lower_farith` replaces `__fadd_*`, `__fsub_*`, `__fmul_*`, and
    `__fmaf_*`. Plain addition and multiplication round ties to even, so
    a value below one ulp disappears. Round-up of 1 plus that value is
    the next float, and a halfway tie stays on the even float. Overflow
    toward zero stays at the finite maximum. Two +0 stay +0 in every
    mode. +0 plus -0 is -0 only when rounding toward -inf. A NaN operand
    becomes the canonical quiet NaN `0x7fffffff`. Fused multiply-add of
    `(1+2^-23)` squared minus the rounded square is `2^-46`, which a
    rounded product plus the addend loses.
    `lower_darith` replaces `__dadd_*`, `__dsub_*`, `__dmul_*`, and
    `__fma_rn`, `__fma_rz`, `__fma_ru`, `__fma_rd`. The double fused
    multiply-add is spelled `__fma_rn`, not `__dfma_rn`. There is no wider
    type than double, so the product is rounded from a 106-bit significand.
    `__dadd_rn` of 1 and `2^-53` stays on 1, and round-up takes the next
    double. Overflow toward zero stays at the finite maximum. `+0` plus
    `-0` is `-0` only when rounding toward -inf. A NaN keeps the second
    operand's sign and payload, and a signaling NaN is quieted. `inf * 0`
    is the negative quiet NaN `0xfff8000000000000`. `__dsub` does not flip
    the sign of a NaN. Fused multiply-add of `(1+2^-52)` squared minus the
    rounded square is `2^-104`, which a rounded product plus the addend loses.
    `lower_fdiv` replaces `__fdiv_*`, `__frcp_*`, `__fsqrt_*`, and
    `__frsqrt_rn`. `__frcp_rn(x)` is `__fdiv_rn(1, x)`. `0/0` and a negative
    square root are the canonical quiet NaN `0x7fffffff`. `sqrt(-0)` stays
    `-0`. A tiny quotient rounds up to the smallest subnormal and stays `0`
    in the other modes. Overflow toward -inf stays at the finite maximum.
    `__frsqrt_rn` of the normal `0x187fffff` is `0x53000000`; the reciprocal
    of the rounded square root is the next float.
    `lower_fmin` replaces `fminf`, `fmaxf`, `fmin`, `fmax`, `nextafterf`,
    `nextafter`, `copysignf`, and `copysign`. `fminf(+0, -0)` is `-0` and
    `fmaxf(+0, -0)` is `+0`. A float NaN loses to a number. Two float NaNs
    become the canonical quiet NaN `0x7fffffff`. `nextafterf(+0, -0)` is
    `-0`, and `nextafterf(+0, +inf)` is the smallest subnormal. A float
    `nextafterf` with a NaN operand is that same canonical NaN. Two double
    NaNs passed to `fmin`, `fmax`, or `nextafter` keep the second operand,
    and a signaling NaN is quieted. One NaN passed to `nextafter` returns
    that NaN, quieted.
    `copysign` keeps the magnitude, including a NaN payload.
    `lower_half` replaces `__float2half`, `__float2half_*`, and
    `__half2float`. `__float2half` is the nearest-even conversion.
    `__float2half_rn(1+2^-11)` stays on 1 and round-up takes the next
    half. A value past 65504 stays at the finite maximum when rounding
    toward zero. A NaN becomes `0x7fff`, with the sign dropped.
    `__half2float` of that NaN is the float `0x7fffffff`. Infinity keeps
    its sign, and `-0` stays `-0`. Integer and double values use the same
    half packing. `__int2half_rn(2049)` stays on 2048 and round-up takes
    2050. `65520` rounds to infinity, and toward zero it stays at the
    finite maximum. A double just above `2^-25` becomes the smallest half
    subnormal; casting that double to float first loses it. A double NaN
    becomes `0x7fff`, and `-0` stays `-0`. `__half2int_rn` of 1.5 is 2
    and toward zero is 1. A half of 32768 stays 32768 in 32 bits and
    saturates to 32767 in a short. A half NaN becomes 0 for 32-bit and
    16-bit results, and the high bit for 64-bit results. `__hadd` and
    `__hadd_rn` are the same nearest-even add. `+0` plus `-0` is `+0`,
    and `-0` plus `-0` stays `-0`. A NaN becomes `0x7fff`. Fused
    multiply-add of `(1+2^-10)` squared minus its rounded square is
    `2^-20`, which a separate product plus addend loses. The saturate
    forms clamp to `[0, 1]`, and a NaN or a negative zero becomes `+0`.
    A direct `__hadd` on `__half` is this add. `__hadd` on `int` stays
    the overflow-free integer average. `__hadd2` and the other packed
    forms run that operation on each lane of a `__half2`. The low lane
    is the low 16 bits. A NaN in one lane stays in that lane.
    `__floats2half2_rn` puts the first float in the low lane.
    `__hne` of a NaN is false, and `__hneu` is true. A true `__heq2`
    lane is `0x3c00`. A true `__heq2_mask` lane is `0xffff`. `__hmax`
    keeps the number when the other value is NaN, and two NaNs become
    `0x7fff`. `__hmax(+0, -0)` is `+0`. `__hmin(+0, -0)` is `-0`.
    `__hneg` and `__habs` of a NaN are `0x7fff`.
    `lower_ptx` replaces one inline `lop3.b32`, `bfe.u32`, `bfe.s32`,
    `bfi.b32`, or `prmt.b32` with an integer helper. The template is one
    instruction, the output constraint is `=r`, and the inputs are `r`.
    `lop3.b32` with immediate `0xF0` copies the first operand, `0xCC` the
    second, and `0xAA` the third. `bfe.s32` of a one-bit field that is set
    is all ones. A start at bit 32 copies bit 31 of the source for
    `bfe.s32` and yields 0 for `bfe.u32`. A length of 0 is 0. `bfi.b32`
    with a start at bit 32 leaves
    the base. `prmt.b32` sign-replicates a selector nibble whose high bit
    is set. `__byte_perm` copies that byte. Any other inline PTX stays
    source.
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
    lower_dp4a: bool
    lower_packed_arith: bool
    lower_packed_minmax: bool
    lower_packed_abs: bool
    lower_byte_perm: bool
    lower_hadd: bool
    lower_mulhi: bool
    lower_sad: bool
    lower_packed_sad: bool
    lower_packed_cmp: bool
    lower_packed_set: bool
    lower_packed_avg: bool
    lower_bitops: bool
    lower_mul24: bool
    lower_round: bool
    lower_saturate: bool
    lower_farith: bool
    lower_darith: bool
    lower_fdiv: bool
    lower_fmin: bool
    lower_half: bool
    lower_ptx: bool


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
        lower_dp4a=True,
        lower_packed_arith=True,
        lower_packed_minmax=True,
        lower_packed_abs=True,
        lower_byte_perm=True,
        lower_hadd=True,
        lower_mulhi=True,
        lower_sad=True,
        lower_packed_sad=True,
        lower_packed_cmp=True,
        lower_packed_set=True,
        lower_packed_avg=True,
        lower_bitops=True,
        lower_mul24=True,
        lower_round=True,
        lower_saturate=True,
        lower_farith=True,
        lower_darith=True,
        lower_fdiv=True,
        lower_fmin=True,
        lower_half=True,
        lower_ptx=True,
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
        lower_dp4a=True,
        lower_packed_arith=True,
        lower_packed_minmax=True,
        lower_packed_abs=True,
        lower_byte_perm=True,
        lower_hadd=True,
        lower_mulhi=True,
        lower_sad=True,
        lower_packed_sad=True,
        lower_packed_cmp=True,
        lower_packed_set=True,
        lower_packed_avg=True,
        lower_bitops=True,
        lower_mul24=True,
        lower_round=True,
        lower_saturate=True,
        lower_farith=True,
        lower_darith=True,
        lower_fdiv=True,
        lower_fmin=True,
        lower_half=True,
        lower_ptx=True,
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
        lower_dp4a=False,
        lower_packed_arith=False,
        lower_packed_minmax=False,
        lower_packed_abs=False,
        lower_byte_perm=False,
        lower_hadd=False,
        lower_mulhi=False,
        lower_sad=False,
        lower_packed_sad=False,
        lower_packed_cmp=False,
        lower_packed_set=False,
        lower_packed_avg=False,
        lower_bitops=False,
        lower_mul24=False,
        lower_round=False,
        lower_saturate=False,
        lower_farith=False,
        lower_darith=False,
        lower_fdiv=False,
        lower_fmin=False,
        lower_half=False,
        lower_ptx=False,
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


# Four-byte dot product. Signed bytes vs unsigned bytes are different programs.
DP4A_HELPER = """
static __device__ int cts_dp4a(int a, int b, int c) {
    int sum = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int aa = static_cast<signed char>(static_cast<unsigned>(a) >> (8 * i));
        int bb = static_cast<signed char>(static_cast<unsigned>(b) >> (8 * i));
        sum += aa * bb;
    }
    return sum;
}

static __device__ unsigned cts_dp4a(unsigned a, unsigned b, unsigned c) {
    unsigned sum = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned aa = (a >> (8 * i)) & 0xffu;
        unsigned bb = (b >> (8 * i)) & 0xffu;
        sum += aa * bb;
    }
    return sum;
}
"""


# Packed 16-bit and 8-bit add/sub. Wrap, signed saturation, unsigned saturation.
def _packed_helper(name: str, lanes: int, mode: str, op: str) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    if mode == "wrap":
        body = f"unsigned lane = (aa {op} bb) & {mask:#x}u;"
    elif mode == "signed":
        limit = (1 << (shift - 1)) - 1
        floor = -(1 << (shift - 1))
        cast = "short" if lanes == 2 else "signed char"
        extract = (
            f"static_cast<{cast}>(a >> ({shift} * i))"
            if lanes == 2
            else f"static_cast<{cast}>((a >> ({shift} * i)) & {mask:#x}u)"
        )
        extract_b = extract.replace("(a >>", "(b >>")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {extract};
        int bb = {extract_b};
        int lane = aa {op} bb;
        if (lane > {limit}) lane = {limit};
        if (lane < {floor}) lane = {floor};
        out |= (static_cast<unsigned>(lane) & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""
    else:
        if op == "+":
            body = f"unsigned lane = aa + bb; if (lane > {mask:#x}u) lane = {mask:#x}u;"
        else:
            body = "unsigned lane = aa < bb ? 0u : aa - bb;"
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        {body}
        out |= lane << ({shift} * i);
    }}
    return out;
}}
"""


PACKED_HELPERS = {
    intrinsic: (f"cts_{intrinsic[2:]}", _packed_helper(f"cts_{intrinsic[2:]}", lanes, mode, op))
    for intrinsic, lanes, mode, op in (
        ("__vadd2", 2, "wrap", "+"),
        ("__vaddss2", 2, "signed", "+"),
        ("__vaddus2", 2, "unsigned", "+"),
        ("__vsub2", 2, "wrap", "-"),
        ("__vsubss2", 2, "signed", "-"),
        ("__vsubus2", 2, "unsigned", "-"),
        ("__vadd4", 4, "wrap", "+"),
        ("__vaddss4", 4, "signed", "+"),
        ("__vaddus4", 4, "unsigned", "+"),
        ("__vsub4", 4, "wrap", "-"),
        ("__vsubss4", 4, "signed", "-"),
        ("__vsubus4", 4, "unsigned", "-"),
    )
}


def _packed_minmax_helper(name: str, lanes: int, signed: bool, which: str) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    op = "<" if which == "min" else ">"
    if signed:
        cast = "short" if lanes == 2 else "signed char"
        extract = (
            f"static_cast<{cast}>(a >> ({shift} * i))"
            if lanes == 2
            else f"static_cast<{cast}>((a >> ({shift} * i)) & {mask:#x}u)"
        )
        extract_b = extract.replace("(a >>", "(b >>")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {extract};
        int bb = {extract_b};
        int lane = aa {op} bb ? aa : bb;
        out |= (static_cast<unsigned>(lane) & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        unsigned lane = aa {op} bb ? aa : bb;
        out |= lane << ({shift} * i);
    }}
    return out;
}}
"""


PACKED_MINMAX_HELPERS = {
    intrinsic: (
        f"cts_{intrinsic[2:]}",
        _packed_minmax_helper(f"cts_{intrinsic[2:]}", lanes, signed, which),
    )
    for intrinsic, lanes, signed, which in (
        ("__vmins2", 2, True, "min"),
        ("__vminu2", 2, False, "min"),
        ("__vmaxs2", 2, True, "max"),
        ("__vmaxu2", 2, False, "max"),
        ("__vmins4", 4, True, "min"),
        ("__vminu4", 4, False, "min"),
        ("__vmaxs4", 4, True, "max"),
        ("__vmaxu4", 4, False, "max"),
    )
}


def _signed_lane(lanes: int, shift: int, mask: int, var: str) -> str:
    cast = "short" if lanes == 2 else "signed char"
    if lanes == 2:
        return f"static_cast<{cast}>({var} >> ({shift} * i))"
    return f"static_cast<{cast}>(({var} >> ({shift} * i)) & {mask:#x}u)"


def _packed_abs_helper(name: str, lanes: int, mode: str) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    floor = -(1 << (shift - 1))
    limit = (1 << (shift - 1)) - 1
    extract = _signed_lane(lanes, shift, mask, "a")
    if mode == "abs":
        mag = "int mag = lane < 0 ? -lane : lane;"
    elif mode == "absss":
        mag = f"int mag = lane == {floor} ? {limit} : (lane < 0 ? -lane : lane);"
    elif mode == "neg":
        mag = "int mag = -lane;"
    elif mode == "negss":
        mag = f"int mag = lane == {floor} ? {limit} : -lane;"
    elif mode == "diffs":
        other = _signed_lane(lanes, shift, mask, "b")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {extract};
        int bb = {other};
        int diff = aa - bb;
        int mag = diff < 0 ? -diff : diff;
        out |= (static_cast<unsigned>(mag) & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""
    else:
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        unsigned mag = aa > bb ? aa - bb : bb - aa;
        out |= mag << ({shift} * i);
    }}
    return out;
}}
"""
    return f"""
static __device__ unsigned {name}(unsigned a) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int lane = {extract};
        {mag}
        out |= (static_cast<unsigned>(mag) & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""


# (replacement, helper, argument count). __vnegss2 matches __vneg2 on CUDA.
PACKED_ABS_HELPERS = {
    intrinsic: (
        f"cts_{intrinsic[2:]}",
        _packed_abs_helper(f"cts_{intrinsic[2:]}", lanes, mode),
        arity,
    )
    for intrinsic, lanes, mode, arity in (
        ("__vabs2", 2, "abs", 1),
        ("__vabsss2", 2, "absss", 1),
        ("__vneg2", 2, "neg", 1),
        ("__vnegss2", 2, "neg", 1),
        ("__vabs4", 4, "abs", 1),
        ("__vabsss4", 4, "absss", 1),
        ("__vneg4", 4, "neg", 1),
        ("__vnegss4", 4, "negss", 1),
        ("__vabsdiffs2", 2, "diffs", 2),
        ("__vabsdiffu2", 2, "diffu", 2),
        ("__vabsdiffs4", 4, "diffs", 2),
        ("__vabsdiffu4", 4, "diffu", 2),
    )
}


def _packed_sad_helper(name: str, lanes: int, signed: bool) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    if signed:
        left = _signed_lane(lanes, shift, mask, "a")
        right = _signed_lane(lanes, shift, mask, "b")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned sum = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {left};
        int bb = {right};
        int diff = aa - bb;
        sum += static_cast<unsigned>(diff < 0 ? -diff : diff);
    }}
    return sum;
}}
"""
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned sum = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        sum += aa >= bb ? aa - bb : bb - aa;
    }}
    return sum;
}}
"""


PACKED_SAD_HELPERS = {
    intrinsic: (
        f"cts_{intrinsic[2:]}",
        _packed_sad_helper(f"cts_{intrinsic[2:]}", lanes, signed),
    )
    for intrinsic, lanes, signed in (
        ("__vsads2", 2, True),
        ("__vsadu2", 2, False),
        ("__vsads4", 4, True),
        ("__vsadu4", 4, False),
    )
}


def _packed_cmp_helper(name: str, lanes: int, op: str, signed: bool) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    if signed:
        left = _signed_lane(lanes, shift, mask, "a")
        right = _signed_lane(lanes, shift, mask, "b")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {left};
        int bb = {right};
        if (aa {op} bb) out |= {mask:#x}u << ({shift} * i);
    }}
    return out;
}}
"""
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        if (aa {op} bb) out |= {mask:#x}u << ({shift} * i);
    }}
    return out;
}}
"""


def _packed_cmp_specs() -> tuple[tuple[str, int, str, bool], ...]:
    specs: list[tuple[str, int, str, bool]] = []
    for lanes in (2, 4):
        specs.append((f"__vcmpeq{lanes}", lanes, "==", False))
        specs.append((f"__vcmpne{lanes}", lanes, "!=", False))
        for rel, op in (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<=")):
            specs.append((f"__vcmp{rel}s{lanes}", lanes, op, True))
            specs.append((f"__vcmp{rel}u{lanes}", lanes, op, False))
    return tuple(specs)


PACKED_CMP_HELPERS = {
    intrinsic: (
        f"cts_{intrinsic[2:]}",
        _packed_cmp_helper(f"cts_{intrinsic[2:]}", lanes, op, signed),
    )
    for intrinsic, lanes, op, signed in _packed_cmp_specs()
}


def _packed_set_helper(name: str, lanes: int, op: str, signed: bool) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    if signed:
        left = _signed_lane(lanes, shift, mask, "a")
        right = _signed_lane(lanes, shift, mask, "b")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {left};
        int bb = {right};
        if (aa {op} bb) out |= 1u << ({shift} * i);
    }}
    return out;
}}
"""
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        if (aa {op} bb) out |= 1u << ({shift} * i);
    }}
    return out;
}}
"""


PACKED_SET_HELPERS = {
    intrinsic.replace("__vcmp", "__vset", 1): (
        f"cts_{intrinsic.replace('__vcmp', '__vset', 1)[2:]}",
        _packed_set_helper(
            f"cts_{intrinsic.replace('__vcmp', '__vset', 1)[2:]}",
            lanes,
            op,
            signed,
        ),
    )
    for intrinsic, lanes, op, signed in _packed_cmp_specs()
}


def _packed_avg_helper(name: str, lanes: int, mode: str) -> str:
    shift = 32 // lanes
    mask = (1 << shift) - 1
    if mode == "savg":
        left = _signed_lane(lanes, shift, mask, "a")
        right = _signed_lane(lanes, shift, mask, "b")
        return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        int aa = {left};
        int bb = {right};
        int sum = aa + bb;
        int avg = sum < 0 ? (sum >> 1) : ((sum + 1) >> 1);
        out |= (static_cast<unsigned>(avg) & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""
    expr = "(aa | bb) - ((aa ^ bb) >> 1)" if mode == "uavg" else "(aa & bb) + ((aa ^ bb) >> 1)"
    return f"""
static __device__ unsigned {name}(unsigned a, unsigned b) {{
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < {lanes}; ++i) {{
        unsigned aa = (a >> ({shift} * i)) & {mask:#x}u;
        unsigned bb = (b >> ({shift} * i)) & {mask:#x}u;
        unsigned avg = {expr};
        out |= (avg & {mask:#x}u) << ({shift} * i);
    }}
    return out;
}}
"""


PACKED_AVG_HELPERS = {
    intrinsic: (
        f"cts_{intrinsic[2:]}",
        _packed_avg_helper(f"cts_{intrinsic[2:]}", lanes, mode),
    )
    for intrinsic, lanes, mode in (
        ("__vavgs2", 2, "savg"),
        ("__vavgs4", 4, "savg"),
        ("__vavgu2", 2, "uavg"),
        ("__vavgu4", 4, "uavg"),
        ("__vhaddu2", 2, "uhadd"),
        ("__vhaddu4", 4, "uhadd"),
    )
}


BIT_HELPERS = {
    "__clz": (
        "cts_clz",
        """
static __device__ int cts_clz(int value) {
    unsigned bits = static_cast<unsigned>(value);
    int count = 0;
    for (int i = 31; i >= 0; --i) {
        if ((bits >> i) & 1u) return count;
        count += 1;
    }
    return 32;
}
""",
    ),
    "__clzll": (
        "cts_clzll",
        """
static __device__ int cts_clzll(long long value) {
    unsigned long long bits = static_cast<unsigned long long>(value);
    int count = 0;
    for (int i = 63; i >= 0; --i) {
        if ((bits >> i) & 1ull) return count;
        count += 1;
    }
    return 64;
}
""",
    ),
    "__ffs": (
        "cts_ffs",
        """
static __device__ int cts_ffs(int value) {
    unsigned bits = static_cast<unsigned>(value);
    if (bits == 0) return 0;
    int count = 1;
    while ((bits & 1u) == 0) {
        bits >>= 1;
        count += 1;
    }
    return count;
}
""",
    ),
    "__ffsll": (
        "cts_ffsll",
        """
static __device__ int cts_ffsll(long long value) {
    unsigned long long bits = static_cast<unsigned long long>(value);
    if (bits == 0) return 0;
    int count = 1;
    while ((bits & 1ull) == 0) {
        bits >>= 1;
        count += 1;
    }
    return count;
}
""",
    ),
    "__popc": (
        "cts_popc",
        """
static __device__ int cts_popc(unsigned value) {
    int count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1;
    }
    return count;
}
""",
    ),
    "__popcll": (
        "cts_popcll",
        """
static __device__ int cts_popcll(unsigned long long value) {
    int count = 0;
    while (value != 0) {
        count += value & 1ull;
        value >>= 1;
    }
    return count;
}
""",
    ),
    "__brev": (
        "cts_brev",
        """
static __device__ unsigned cts_brev(unsigned value) {
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) out |= ((value >> i) & 1u) << (31 - i);
    return out;
}
""",
    ),
    "__brevll": (
        "cts_brevll",
        """
static __device__ unsigned long long cts_brevll(unsigned long long value) {
    unsigned long long out = 0;
#pragma unroll
    for (int i = 0; i < 64; ++i) out |= ((value >> i) & 1ull) << (63 - i);
    return out;
}
""",
    ),
}


MUL24_HELPERS = {
    "__mul24": (
        "cts_mul24",
        """
static __device__ int cts_mul24(int x, int y) {
    int a = x & 0xffffff;
    int b = y & 0xffffff;
    if (a & 0x800000) a -= 0x1000000;
    if (b & 0x800000) b -= 0x1000000;
    return static_cast<int>(static_cast<long long>(a) * static_cast<long long>(b));
}
""",
    ),
    "__umul24": (
        "cts_umul24",
        """
static __device__ unsigned cts_umul24(unsigned x, unsigned y) {
    return (x & 0xffffffu) * (y & 0xffffffu);
}
""",
    ),
}


SAD_HELPERS = {
    "__sad": (
        "cts_sad",
        """
static __device__ unsigned cts_sad(int x, int y, unsigned z) {
    long long diff = static_cast<long long>(x) - static_cast<long long>(y);
    if (diff < 0) diff = -diff;
    return static_cast<unsigned>(static_cast<unsigned long long>(diff) + z);
}
""",
    ),
    "__usad": (
        "cts_usad",
        """
static __device__ unsigned cts_usad(unsigned x, unsigned y, unsigned z) {
    unsigned diff = x >= y ? x - y : y - x;
    return diff + z;
}
""",
    ),
}


# __mul64hi calls cts_umul64hi, so the unsigned 64-bit helper is emitted before it.
MULHI_HELPERS = {
    "__mulhi": (
        "cts_mulhi",
        """
static __device__ int cts_mulhi(int x, int y) {
    return static_cast<int>((static_cast<long long>(x) * static_cast<long long>(y)) >> 32);
}
""",
    ),
    "__umulhi": (
        "cts_umulhi",
        """
static __device__ unsigned cts_umulhi(unsigned x, unsigned y) {
    return static_cast<unsigned>((static_cast<unsigned long long>(x) * y) >> 32);
}
""",
    ),
    "__umul64hi": (
        "cts_umul64hi",
        """
static __device__ unsigned long long cts_umul64hi(unsigned long long x, unsigned long long y) {
    unsigned long long x0 = x & 0xffffffffull;
    unsigned long long x1 = x >> 32;
    unsigned long long y0 = y & 0xffffffffull;
    unsigned long long y1 = y >> 32;
    unsigned long long p00 = x0 * y0;
    unsigned long long p01 = x0 * y1;
    unsigned long long p10 = x1 * y0;
    unsigned long long p11 = x1 * y1;
    unsigned long long mid = (p00 >> 32) + (p01 & 0xffffffffull) + (p10 & 0xffffffffull);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}
""",
    ),
    "__mul64hi": (
        "cts_mul64hi",
        """
static __device__ long long cts_mul64hi(long long x, long long y) {
    unsigned long long hi = cts_umul64hi(static_cast<unsigned long long>(x),
                                         static_cast<unsigned long long>(y));
    if (x < 0) hi -= static_cast<unsigned long long>(y);
    if (y < 0) hi -= static_cast<unsigned long long>(x);
    return static_cast<long long>(hi);
}
""",
    ),
}


HADD_HELPERS = {
    "__hadd": (
        "cts_hadd",
        """
static __device__ int cts_hadd(int x, int y) {
    return (x & y) + ((x ^ y) >> 1);
}
""",
    ),
    "__rhadd": (
        "cts_rhadd",
        """
static __device__ int cts_rhadd(int x, int y) {
    return (x | y) - ((x ^ y) >> 1);
}
""",
    ),
    "__uhadd": (
        "cts_uhadd",
        """
static __device__ unsigned cts_uhadd(unsigned x, unsigned y) {
    return (x & y) + ((x ^ y) >> 1);
}
""",
    ),
    "__urhadd": (
        "cts_urhadd",
        """
static __device__ unsigned cts_urhadd(unsigned x, unsigned y) {
    return (x | y) - ((x ^ y) >> 1);
}
""",
    ),
}


BYTE_PERM_HELPER = """
static __device__ unsigned cts_byte_perm(unsigned x, unsigned y, unsigned selector) {
    unsigned long long source = (static_cast<unsigned long long>(y) << 32) | x;
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned index = (selector >> (4 * i)) & 7u;
        unsigned byte = static_cast<unsigned>(source >> (8 * index)) & 0xffu;
        out |= byte << (8 * i);
    }
    return out;
}
"""


# Measured on sm_70. __float2int_* turns NaN into 0 and saturates past the
# signed range. __float2ll_* and __float2ull_* turn NaN into the high bit.
# Every __double2int_* mode turns NaN into 0x80000000. __double2float_*
# turns a NaN into a quiet float NaN with the same sign. Every 32-bit
# integer is exact in double, so int-to-float rounds that double down to
# float. A long long past 2^53 is not exact in double, so the 64-bit path
# rounds the bits.
_ROUND_CALL = {"rn": "rintf", "rz": "truncf", "ru": "ceilf", "rd": "floorf"}


def _float_to_int(mode: str, unsigned: bool) -> str:
    call = _ROUND_CALL[mode]
    if not unsigned:
        name = f"cts_float2int_{mode}"
        return f"""
static __device__ int {name}(float x) {{
    float y = {call}(x);
    if (!(y == y)) return 0;
    if (y >= 2147483648.0f) return 2147483647;
    if (y < -2147483648.0f) return (int)0x80000000;
    return (int)y;
}}
"""
    name = f"cts_float2uint_{mode}"
    return f"""
static __device__ unsigned {name}(float x) {{
    float y = {call}(x);
    if (!(y == y) || y <= 0.0f) return 0u;
    if (y >= 4294967296.0f) return 0xffffffffu;
    return (unsigned)y;
}}
"""


def _float_to_ll(mode: str, unsigned: bool) -> str:
    call = _ROUND_CALL[mode]
    if not unsigned:
        name = f"cts_float2ll_{mode}"
        return f"""
static __device__ long long {name}(float x) {{
    float y = {call}(x);
    if (!(y == y)) return (long long)0x8000000000000000ULL;
    if (y >= 0x1p63f) return 9223372036854775807LL;
    if (y <= -0x1p63f) return (long long)0x8000000000000000ULL;
    return (long long)y;
}}
"""
    name = f"cts_float2ull_{mode}"
    return f"""
static __device__ unsigned long long {name}(float x) {{
    float y = {call}(x);
    if (!(y == y)) return 0x8000000000000000ull;
    if (y <= 0.0f) return 0ull;
    if (y >= 0x1p64f) return 0xffffffffffffffffull;
    return (unsigned long long)y;
}}
"""


def _to_float(prefix: str, ctype: str, callee: str, mode: str, code: int) -> str:
    name = f"cts_{prefix}_{mode}"
    return f"""
static __device__ float {name}({ctype} x) {{
    return {callee}(x, {code});
}}
"""


ROUND_BITS = """
static __device__ unsigned cts_round_fbits(float value) {
    union {
        float as_float;
        unsigned as_bits;
    } converted;
    converted.as_float = value;
    return converted.as_bits;
}

static __device__ float cts_round_ffrom(unsigned bits) {
    union {
        unsigned as_bits;
        float as_float;
    } converted;
    converted.as_bits = bits;
    return converted.as_float;
}

static __device__ float cts_round_adj(float value, int plus) {
    unsigned bits = cts_round_fbits(value);
    return cts_round_ffrom(plus ? bits + 1u : bits - 1u);
}
"""


ROUND_I32 = """
static __device__ float cts_round_i32(int x, int mode) {
    double exact = (double)x;
    float f = (float)exact;
    if ((double)f == exact || mode == 0) return f;
    int above = (double)f > exact;
    if (x >= 0) {
        if (mode == 2) return above ? f : cts_round_adj(f, 1);
        return above ? cts_round_adj(f, 0) : f;
    }
    if (mode == 3) return above ? cts_round_adj(f, 1) : f;
    return above ? f : cts_round_adj(f, 0);
}

static __device__ float cts_round_u32(unsigned x, int mode) {
    double exact = (double)x;
    float f = (float)exact;
    if ((double)f == exact || mode == 0) return f;
    int above = (double)f > exact;
    if (mode == 2) return above ? f : cts_round_adj(f, 1);
    return above ? cts_round_adj(f, 0) : f;
}
"""


ROUND_I64 = """
static __device__ float cts_round_u64(unsigned long long x, int mode) {
    if (x == 0ull) return 0.0f;
    int clz = 0;
    unsigned long long t = x;
    while ((t >> 63) == 0ull) {
        t <<= 1;
        ++clz;
    }
    int exp = 63 - clz;
    if (exp <= 23) return (float)x;
    int shift = exp - 23;
    unsigned long long dropped = x & ((1ull << shift) - 1ull);
    unsigned long long mant = x >> shift;
    unsigned long long half = 1ull << (shift - 1);
    int up = 0;
    if (mode == 0) {
        if (dropped > half || (dropped == half && (mant & 1ull))) up = 1;
    } else if (mode == 2) {
        if (dropped) up = 1;
    }
    if (up) {
        mant += 1ull;
        if (mant == (1ull << 24)) {
            mant >>= 1;
            ++exp;
        }
    }
    unsigned bits = ((unsigned)(exp + 127) << 23) | (unsigned)(mant & 0x7fffffull);
    return cts_round_ffrom(bits);
}

static __device__ float cts_round_s64(long long x, int mode) {
    if (x >= 0) return cts_round_u64((unsigned long long)x, mode);
    unsigned long long mag = (unsigned long long)(-(x + 1)) + 1ull;
    int mag_mode = mode;
    if (mode == 2) mag_mode = 1;
    else if (mode == 3) mag_mode = 2;
    return -cts_round_u64(mag, mag_mode);
}
"""


# Measured on sm_70. Every 32-bit integer is exact in double, so the rn-only
# int and uint conversions are casts. A 64-bit magnitude past 2^53 rounds
# from its bits. 2^53+1 is the tie below the odd neighbor and stays on 2^53
# in rn. 2^53+3 is the tie above it and rounds to the even neighbor.
# Negating swaps ru with a toward-zero magnitude and rd with a round-up
# magnitude. The maximum unsigned value is one below 2^64, so rz stops at
# 2^64-2048 and ru reaches 2^64.
I2D_U64 = """
static __device__ unsigned long long cts_i2d_u64(unsigned long long x, int mode) {
    if (x == 0ull) return 0ull;
    int clz = 0;
    unsigned long long top = x;
    while ((top >> 63) == 0ull) {
        top <<= 1;
        ++clz;
    }
    int exp = 63 - clz;
    int shift = exp - 52;
    unsigned long long dropped = 0ull;
    unsigned long long mant = x;
    if (shift > 0) {
        dropped = x & ((1ull << shift) - 1ull);
        mant = x >> shift;
    } else if (shift < 0) {
        mant = x << (unsigned)(-shift);
    }
    int up = 0;
    if (shift > 0) {
        unsigned long long half = 1ull << (shift - 1);
        if (mode == 0) {
            if (dropped > half || (dropped == half && (mant & 1ull))) up = 1;
        } else if (mode == 2 && dropped) {
            up = 1;
        }
    }
    if (up) {
        mant += 1ull;
        if (mant == (1ull << 53)) {
            mant >>= 1;
            ++exp;
        }
    }
    return ((unsigned long long)(exp + 1023) << 52) | (mant & ((1ull << 52) - 1ull));
}
"""

I2D_S64 = """
static __device__ unsigned long long cts_i2d_s64(long long x, int mode) {
    if (x >= 0) return cts_i2d_u64((unsigned long long)x, mode);
    unsigned long long mag = (unsigned long long)(-(x + 1)) + 1ull;
    int mag_mode = mode;
    if (mode == 2) mag_mode = 1;
    else if (mode == 3) mag_mode = 2;
    return cts_i2d_u64(mag, mag_mode) | 0x8000000000000000ull;
}
"""


def _i2d_exact(stem: str, ctype: str) -> str:
    return f"""
static __device__ double cts_{stem}_rn({ctype} x) {{
    return (double)x;
}}
"""


def _i2d_wrap(stem: str, ctype: str, core: str, mode: str) -> str:
    return f"""
static __device__ double cts_{stem}_{mode}({ctype} x) {{
    union {{
        unsigned long long as_bits;
        double as_double;
    }} out;
    out.as_bits = {core}(x, {_MODE_CODE[mode]});
    return out.as_double;
}}
"""


_DOUBLE_RZ = {
    "__double2int_rz": (
        "cts_double2int_rz",
        """
static __device__ int cts_double2int_rz(double x) {
    if (!(x == x)) return (int)0x80000000;
    if (x >= 2147483648.0) return 2147483647;
    if (x < -2147483648.0) return (int)0x80000000;
    return (int)x;
}
""",
    ),
    "__double2uint_rz": (
        "cts_double2uint_rz",
        """
static __device__ unsigned cts_double2uint_rz(double x) {
    if (!(x == x)) return 0x80000000u;
    if (x <= 0.0) return 0u;
    if (x >= 4294967296.0) return 0xffffffffu;
    return (unsigned)x;
}
""",
    ),
    "__double2ll_rz": (
        "cts_double2ll_rz",
        """
static __device__ long long cts_double2ll_rz(double x) {
    if (!(x == x)) return (long long)0x8000000000000000ULL;
    if (x >= 0x1p63) return 9223372036854775807LL;
    if (x <= -0x1p63) return (long long)0x8000000000000000ULL;
    return (long long)x;
}
""",
    ),
    "__double2ull_rz": (
        "cts_double2ull_rz",
        """
static __device__ unsigned long long cts_double2ull_rz(double x) {
    if (!(x == x)) return 0x8000000000000000ull;
    if (x <= 0.0) return 0ull;
    if (x >= 0x1p64) return 0xffffffffffffffffull;
    return (unsigned long long)x;
}
""",
    ),
}

_MODE_CODE = {"rn": 0, "rz": 1, "ru": 2, "rd": 3}

ROUND_HELPERS: dict[str, tuple[str, str]] = {}
for _mode in ("rn", "rz", "ru", "rd"):
    ROUND_HELPERS[f"__float2int_{_mode}"] = (f"cts_float2int_{_mode}", _float_to_int(_mode, False))
    ROUND_HELPERS[f"__float2uint_{_mode}"] = (f"cts_float2uint_{_mode}", _float_to_int(_mode, True))
    ROUND_HELPERS[f"__int2float_{_mode}"] = (
        f"cts_int2float_{_mode}",
        _to_float("int2float", "int", "cts_round_i32", _mode, _MODE_CODE[_mode]),
    )
    ROUND_HELPERS[f"__uint2float_{_mode}"] = (
        f"cts_uint2float_{_mode}",
        _to_float("uint2float", "unsigned", "cts_round_u32", _mode, _MODE_CODE[_mode]),
    )
    ROUND_HELPERS[f"__float2ll_{_mode}"] = (f"cts_float2ll_{_mode}", _float_to_ll(_mode, False))
    ROUND_HELPERS[f"__float2ull_{_mode}"] = (f"cts_float2ull_{_mode}", _float_to_ll(_mode, True))
    ROUND_HELPERS[f"__ll2float_{_mode}"] = (
        f"cts_ll2float_{_mode}",
        _to_float("ll2float", "long long", "cts_round_s64", _mode, _MODE_CODE[_mode]),
    )
    ROUND_HELPERS[f"__ull2float_{_mode}"] = (
        f"cts_ull2float_{_mode}",
        _to_float("ull2float", "unsigned long long", "cts_round_u64", _mode, _MODE_CODE[_mode]),
    )
    ROUND_HELPERS[f"__ll2double_{_mode}"] = (
        f"cts_ll2double_{_mode}",
        _i2d_wrap("ll2double", "long long", "cts_i2d_s64", _mode),
    )
    ROUND_HELPERS[f"__ull2double_{_mode}"] = (
        f"cts_ull2double_{_mode}",
        _i2d_wrap("ull2double", "unsigned long long", "cts_i2d_u64", _mode),
    )
ROUND_HELPERS["__int2double_rn"] = ("cts_int2double_rn", _i2d_exact("int2double", "int"))
ROUND_HELPERS["__uint2double_rn"] = ("cts_uint2double_rn", _i2d_exact("uint2double", "unsigned"))
ROUND_HELPERS.update(_DOUBLE_RZ)


def round_needs_bits(names: set[str]) -> bool:
    return any(
        name.startswith("cts_int2float_")
        or name.startswith("cts_uint2float_")
        or name.startswith("cts_ll2float_")
        or name.startswith("cts_ull2float_")
        for name in names
    )


def round_needs_i32(names: set[str]) -> bool:
    return any(name.startswith("cts_int2float_") or name.startswith("cts_uint2float_") for name in names)


def round_needs_i64(names: set[str]) -> bool:
    return any(name.startswith("cts_ll2float_") or name.startswith("cts_ull2float_") for name in names)


def round_needs_i2d_u64(names: set[str]) -> bool:
    return any(name.startswith(("cts_ll2double_", "cts_ull2double_")) for name in names)


def round_needs_i2d_s64(names: set[str]) -> bool:
    return any(name.startswith("cts_ll2double_") for name in names)


def round_needs_i2d_i32(names: set[str]) -> bool:
    return any(name.startswith(("cts_int2double_", "cts_uint2double_")) for name in names)

D2F_CORE = """
static __device__ unsigned cts_d2f_pack(int sign, int exp, unsigned long long mag, int mode) {
    if (mag == 0) return sign ? 0x80000000u : 0u;
    unsigned long long top = mag;
    while (top >= (1ull << 56)) { top >>= 1; exp++; }
    while (top < (1ull << 55) && exp > 1) { top <<= 1; exp--; }
    if (exp < 1 || (exp == 1 && top < (1ull << 55))) {
        int need = 1 - exp;
        exp = 0;
        if (need >= 64) top = top ? 1 : 0;
        else if (need) {
            unsigned long long lost = top & ((1ull << need) - 1);
            top >>= need;
            if (lost) top |= 1;
        }
    }
    unsigned long long rem = top & 0xffffffffull;
    unsigned sig = (unsigned)(top >> 32) & 0xffffffu;
    int toward_inf = sign ? (mode == 3) : (mode == 2);
    int up = 0;
    if (mode == 0) up = rem > 0x80000000ull || (rem == 0x80000000ull && (sig & 1u));
    else if (toward_inf) up = rem != 0;
    if (up) {
        sig += 1u;
        if (exp == 0) {
            if (sig == 0x800000u) exp = 1;
        } else if (sig == 0x1000000u) {
            sig = 0x800000u;
            exp++;
        }
    }
    if (exp >= 255) {
        int to_inf = (mode == 0) || (sign ? mode == 3 : mode == 2);
        if (to_inf) return ((unsigned)sign << 31) | 0x7f800000u;
        return ((unsigned)sign << 31) | 0x7f7fffffu;
    }
    if (exp <= 0) return ((unsigned)sign << 31) | (sig & 0x7fffffu);
    return ((unsigned)sign << 31) | ((unsigned)exp << 23) | (sig & 0x7fffffu);
}

static __device__ float cts_d2f(double value, int mode) {
    union { double as_double; unsigned long long as_bits; } in;
    in.as_double = value;
    unsigned sign = (unsigned)(in.as_bits >> 63);
    int exp = (int)((in.as_bits >> 52) & 0x7ff);
    unsigned long long frac = in.as_bits & 0xfffffffffffffULL;
    if (exp == 0x7ff) {
        union { float as_float; unsigned as_bits; } out;
        out.as_bits = frac ? (0x7fc00000u | (sign << 31)) : ((sign << 31) | 0x7f800000u);
        return out.as_float;
    }
    unsigned long long sig = exp ? (frac | (1ULL << 52)) : frac;
    int fexp = exp ? (exp - 896) : -895;
    union { float as_float; unsigned as_bits; } out;
    out.as_bits = cts_d2f_pack((int)sign, fexp, sig << 3, mode);
    return out.as_float;
}
"""

_DOUBLE_INT_CALL = {"rn": "rint", "ru": "ceil", "rd": "floor"}


def _double_to_float(mode: str) -> str:
    name = f"cts_double2float_{mode}"
    code = _MODE_CODE[mode]
    return f"""
static __device__ float {name}(double x) {{
    return cts_d2f(x, {code});
}}
"""


def _double_to_int(mode: str) -> str:
    call = _DOUBLE_INT_CALL[mode]
    name = f"cts_double2int_{mode}"
    return f"""
static __device__ int {name}(double x) {{
    if (!(x == x)) return (int)0x80000000;
    double y = {call}(x);
    if (y >= 2147483648.0) return 2147483647;
    if (y < -2147483648.0) return (int)0x80000000;
    return (int)y;
}}
"""


def _double_to_uint(mode: str) -> str:
    call = _DOUBLE_INT_CALL[mode]
    name = f"cts_double2uint_{mode}"
    return f"""
static __device__ unsigned {name}(double x) {{
    if (!(x == x)) return 0x80000000u;
    double y = {call}(x);
    if (y <= 0.0) return 0u;
    if (y >= 4294967296.0) return 0xffffffffu;
    return (unsigned)y;
}}
"""


def _double_to_ll(mode: str) -> str:
    call = _DOUBLE_INT_CALL[mode]
    name = f"cts_double2ll_{mode}"
    return f"""
static __device__ long long {name}(double x) {{
    if (!(x == x)) return (long long)0x8000000000000000ULL;
    double y = {call}(x);
    if (y >= 0x1p63) return 9223372036854775807LL;
    if (y <= -0x1p63) return (long long)0x8000000000000000ULL;
    return (long long)y;
}}
"""


def _double_to_ull(mode: str) -> str:
    call = _DOUBLE_INT_CALL[mode]
    name = f"cts_double2ull_{mode}"
    return f"""
static __device__ unsigned long long {name}(double x) {{
    if (!(x == x)) return 0x8000000000000000ull;
    double y = {call}(x);
    if (y <= 0.0) return 0ull;
    if (y >= 0x1p64) return 0xffffffffffffffffull;
    return (unsigned long long)y;
}}
"""


for _mode in ("rn", "rz", "ru", "rd"):
    ROUND_HELPERS[f"__double2float_{_mode}"] = (f"cts_double2float_{_mode}", _double_to_float(_mode))
for _mode in ("rn", "ru", "rd"):
    ROUND_HELPERS[f"__double2int_{_mode}"] = (f"cts_double2int_{_mode}", _double_to_int(_mode))
    ROUND_HELPERS[f"__double2uint_{_mode}"] = (f"cts_double2uint_{_mode}", _double_to_uint(_mode))
    ROUND_HELPERS[f"__double2ll_{_mode}"] = (f"cts_double2ll_{_mode}", _double_to_ll(_mode))
    ROUND_HELPERS[f"__double2ull_{_mode}"] = (f"cts_double2ull_{_mode}", _double_to_ull(_mode))


def round_needs_d2f(names: set[str]) -> bool:
    return any(name.startswith("cts_double2float_") for name in names)




SATURATE_HELPERS = {
    "__saturatef": (
        "cts_saturatef",
        """
static __device__ float cts_saturatef(float x) {
    if (!(x == x) || x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x;
}
""",
    ),
}



# The 24-bit significand LSB sits at bit 32. Bits below that are the remainder.
FARITH_PACK = """
static __device__ unsigned cts_fpack(int sign, int exp, unsigned long long mag, int mode) {
    if (mag == 0) return sign ? 0x80000000u : 0u;
    unsigned long long top = mag;
    while (top >= (1ull << 56)) { top >>= 1; exp++; }
    while (top < (1ull << 55) && exp > 1) { top <<= 1; exp--; }
    if (exp < 1 || (exp == 1 && top < (1ull << 55))) {
        int need = 1 - exp;
        exp = 0;
        if (need >= 64) top = top ? 1 : 0;
        else if (need) {
            unsigned long long lost = top & ((1ull << need) - 1);
            top >>= need;
            if (lost) top |= 1;
        }
    }
    unsigned long long rem = top & 0xffffffffull;
    unsigned sig = (unsigned)(top >> 32) & 0xffffffu;
    int toward_inf = sign ? (mode == 3) : (mode == 2);
    int up = 0;
    if (mode == 0) up = rem > 0x80000000ull || (rem == 0x80000000ull && (sig & 1u));
    else if (toward_inf) up = rem != 0;
    if (up) {
        sig += 1u;
        if (exp == 0) {
            if (sig == 0x800000u) exp = 1;
        } else if (sig == 0x1000000u) {
            sig = 0x800000u;
            exp++;
        }
    }
    if (exp >= 255) {
        int to_inf = (mode == 0) || (sign ? mode == 3 : mode == 2);
        if (to_inf) return ((unsigned)sign << 31) | 0x7f800000u;
        return ((unsigned)sign << 31) | 0x7f7fffffu;
    }
    if (exp <= 0) return ((unsigned)sign << 31) | (sig & 0x7fffffu);
    return ((unsigned)sign << 31) | ((unsigned)exp << 23) | (sig & 0x7fffffu);
}
"""

FARITH_ADD = """
static __device__ unsigned cts_fadd(unsigned a, unsigned b, int mode) {
    unsigned sa = a >> 31, sb = b >> 31;
    int ea = (a >> 23) & 0xff, eb = (b >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu, fb = b & 0x7fffffu;
    if (ea == 255 || eb == 255) {
        if ((ea == 255 && fa) || (eb == 255 && fb)) return 0x7fffffffu;
        if (ea == 255 && eb == 255) return sa == sb ? ((sa << 31) | 0x7f800000u) : 0x7fffffffu;
        return ea == 255 ? ((sa << 31) | 0x7f800000u) : ((sb << 31) | 0x7f800000u);
    }
    unsigned long long ma, mb;
    int aexp, bexp;
    if (ea == 0) { ma = (unsigned long long)fa << 32; aexp = 1; }
    else { ma = (unsigned long long)(fa | 0x800000u) << 32; aexp = ea; }
    if (eb == 0) { mb = (unsigned long long)fb << 32; bexp = 1; }
    else { mb = (unsigned long long)(fb | 0x800000u) << 32; bexp = eb; }
    if (aexp < bexp || (aexp == bexp && ma < mb)) {
        unsigned ts = sa; sa = sb; sb = ts;
        int te = aexp; aexp = bexp; bexp = te;
        unsigned long long tm = ma; ma = mb; mb = tm;
    }
    int sh = aexp - bexp;
    if (sh >= 63) mb = mb ? 1ull : 0ull;
    else if (sh) {
        unsigned long long lost = mb & ((1ull << sh) - 1ull);
        mb >>= sh;
        if (lost) mb |= 1ull;
    }
    if (sa == sb) return cts_fpack((int)sa, aexp, ma + mb, mode);
    if (ma == mb) return (mode == 3) ? 0x80000000u : 0u;
    return cts_fpack((int)sa, aexp, ma - mb, mode);
}
"""

FARITH_MUL = """
static __device__ unsigned cts_fmul(unsigned a, unsigned b, int mode) {
    unsigned sa = a >> 31, sb = b >> 31;
    int sign = (int)(sa ^ sb);
    int ea = (a >> 23) & 0xff, eb = (b >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu, fb = b & 0x7fffffu;
    if (ea == 255 || eb == 255) {
        if ((ea == 255 && fa) || (eb == 255 && fb)) return 0x7fffffffu;
        int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
        if ((ea == 255 && b0) || (eb == 255 && a0)) return 0x7fffffffu;
        return ((unsigned)sign << 31) | 0x7f800000u;
    }
    if ((ea | fa) == 0 || (eb | fb) == 0) return sign ? 0x80000000u : 0u;
    unsigned long long ma = ea ? (fa | 0x800000u) : fa;
    unsigned long long mb = eb ? (fb | 0x800000u) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 127;
    unsigned long long mag = (ma * mb) << 9;
    return cts_fpack(sign, exp, mag, mode);
}
"""

FARITH_FMA = """
static __device__ unsigned cts_ffma(unsigned a, unsigned b, unsigned c, int mode) {
    unsigned sa = a >> 31, sb = b >> 31, sc = c >> 31;
    int signp = (int)(sa ^ sb);
    int ea = (a >> 23) & 0xff, eb = (b >> 23) & 0xff, ec = (c >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu, fb = b & 0x7fffffu, fc = c & 0x7fffffu;
    if ((ea == 255 && fa) || (eb == 255 && fb) || (ec == 255 && fc)) return 0x7fffffffu;
    int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
    int ainf = ea == 255, binf = eb == 255, cinf = ec == 255;
    if ((ainf && b0) || (binf && a0)) return 0x7fffffffu;
    if (ainf || binf) {
        if (cinf && sc != (unsigned)signp) return 0x7fffffffu;
        return ((unsigned)signp << 31) | 0x7f800000u;
    }
    if (a0 || b0) return cts_fadd(signp ? 0x80000000u : 0u, c, mode);
    unsigned long long pa = ea ? (fa | 0x800000u) : fa;
    unsigned long long pb = eb ? (fb | 0x800000u) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 127;
    unsigned long long mag = (pa * pb) << 9;
    if (mag) {
        while (mag >= (1ull << 56)) { mag >>= 1; exp++; }
        while (mag < (1ull << 55) && exp > 1) { mag <<= 1; exp--; }
    }
    if ((ec | fc) == 0) return cts_fpack(signp, exp, mag, mode);
    unsigned long long mc = ec ? ((unsigned long long)(fc | 0x800000u) << 32) : ((unsigned long long)fc << 32);
    int cexp = ec ? ec : 1;
    int sh = exp - cexp;
    unsigned long long acc = mag;
    if (sh >= 0) {
        if (sh >= 63) mc = mc ? 1ull : 0ull;
        else if (sh) {
            unsigned long long lost = mc & ((1ull << sh) - 1ull);
            mc >>= sh;
            if (lost) mc |= 1ull;
        }
    } else {
        int shc = -sh;
        if (shc >= 63) acc = acc ? 1ull : 0ull;
        else {
            unsigned long long lost = acc & ((1ull << shc) - 1ull);
            acc >>= shc;
            if (lost) acc |= 1ull;
        }
        exp = cexp;
    }
    if ((unsigned)signp == sc) return cts_fpack(signp, exp, acc + mc, mode);
    if (acc == mc) return mode == 3 ? 0x80000000u : 0u;
    if (acc < mc) return cts_fpack((int)sc, exp, mc - acc, mode);
    return cts_fpack(signp, exp, acc - mc, mode);
}
"""

_FARITH_CODE = {"rn": 0, "rz": 1, "ru": 2, "rd": 3}


def _farith_binary(kind: str, mode: str) -> str:
    code = _FARITH_CODE[mode]
    if kind == "sub":
        call = f"cts_fadd(x.as_bits, y.as_bits ^ 0x80000000u, {code})"
    elif kind == "mul":
        call = f"cts_fmul(x.as_bits, y.as_bits, {code})"
    else:
        call = f"cts_fadd(x.as_bits, y.as_bits, {code})"
    return f"""
static __device__ float cts_f{kind}_{mode}(float a, float b) {{
    union {{ float as_float; unsigned as_bits; }} x, y, out;
    x.as_float = a;
    y.as_float = b;
    out.as_bits = {call};
    return out.as_float;
}}
"""


def _farith_fma(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ float cts_fmaf_{mode}(float a, float b, float c) {{
    union {{ float as_float; unsigned as_bits; }} x, y, z, out;
    x.as_float = a;
    y.as_float = b;
    z.as_float = c;
    out.as_bits = cts_ffma(x.as_bits, y.as_bits, z.as_bits, {code});
    return out.as_float;
}}
"""


FARITH_HELPERS: dict[str, tuple[str, str]] = {}
for _kind in ("add", "sub", "mul"):
    for _mode in ("rn", "rz", "ru", "rd"):
        FARITH_HELPERS[f"__f{_kind}_{_mode}"] = (f"cts_f{_kind}_{_mode}", _farith_binary(_kind, _mode))
for _mode in ("rn", "rz", "ru", "rd"):
    FARITH_HELPERS[f"__fmaf_{_mode}"] = (f"cts_fmaf_{_mode}", _farith_fma(_mode))


def farith_needs_add(names: set[str]) -> bool:
    return any(
        name.startswith("cts_fadd_") or name.startswith("cts_fsub_") or name.startswith("cts_fmaf_")
        for name in names
    )


def farith_needs_mul(names: set[str]) -> bool:
    return any(name.startswith("cts_fmul_") for name in names)


def farith_needs_fma(names: set[str]) -> bool:
    return any(name.startswith("cts_fmaf_") for name in names)


DARITH_PACK = """
static __device__ int cts_disnan(unsigned long long x) {
    return ((x >> 52) & 0x7ff) == 0x7ff && (x & 0xfffffffffffffull);
}

static __device__ unsigned long long cts_dquiet(unsigned long long x) {
    return x | 0x0008000000000000ull;
}

static __device__ unsigned long long cts_dpack(int sign, int exp, unsigned __int128 mag, int mode) {
    if (mag == 0) return sign ? 0x8000000000000000ull : 0ull;
    while (mag >= ((unsigned __int128)1 << 117)) { mag >>= 1; exp++; }
    while (mag < ((unsigned __int128)1 << 116) && exp > 1) { mag <<= 1; exp--; }
    if (exp < 1 || (exp == 1 && mag < ((unsigned __int128)1 << 116))) {
        int need = 1 - exp;
        exp = 0;
        if (need >= 128) mag = mag ? 1 : 0;
        else if (need) {
            unsigned __int128 lost = mag & (((unsigned __int128)1 << need) - 1);
            mag >>= need;
            if (lost) mag |= 1;
        }
    }
    unsigned long long rem = (unsigned long long)mag;
    unsigned long long sig = (unsigned long long)(mag >> 64) & 0x1fffffffffffffull;
    int toward_inf = sign ? (mode == 3) : (mode == 2);
    int up = 0;
    if (mode == 0) up = rem > 0x8000000000000000ull || (rem == 0x8000000000000000ull && (sig & 1ull));
    else if (toward_inf) up = rem != 0;
    if (up) {
        sig += 1ull;
        if (exp == 0) {
            if (sig == 0x10000000000000ull) exp = 1;
        } else if (sig == 0x20000000000000ull) {
            sig = 0x10000000000000ull;
            exp++;
        }
    }
    if (exp >= 2047) {
        int to_inf = (mode == 0) || (sign ? mode == 3 : mode == 2);
        if (to_inf) return ((unsigned long long)sign << 63) | 0x7ff0000000000000ull;
        return ((unsigned long long)sign << 63) | 0x7fefffffffffffffull;
    }
    if (exp <= 0) return ((unsigned long long)sign << 63) | (sig & 0xfffffffffffffull);
    return ((unsigned long long)sign << 63) | ((unsigned long long)exp << 52) | (sig & 0xfffffffffffffull);
}
"""

DARITH_ADD = """
static __device__ unsigned long long cts_dadd(unsigned long long a, unsigned long long b, int mode) {
    if (cts_disnan(a) || cts_disnan(b)) return cts_dquiet(cts_disnan(b) ? b : a);
    unsigned long long sa = a >> 63, sb = b >> 63;
    int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff);
    unsigned long long fa = a & 0xfffffffffffffull, fb = b & 0xfffffffffffffull;
    if (ea == 2047 || eb == 2047) {
        if (ea == 2047 && eb == 2047) {
            return sa == sb ? ((sa << 63) | 0x7ff0000000000000ull) : 0xfff8000000000000ull;
        }
        return ea == 2047 ? ((sa << 63) | 0x7ff0000000000000ull) : ((sb << 63) | 0x7ff0000000000000ull);
    }
    unsigned __int128 ma, mb;
    int aexp, bexp;
    if (ea == 0) { ma = (unsigned __int128)fa << 64; aexp = 1; }
    else { ma = (unsigned __int128)(fa | 0x10000000000000ull) << 64; aexp = ea; }
    if (eb == 0) { mb = (unsigned __int128)fb << 64; bexp = 1; }
    else { mb = (unsigned __int128)(fb | 0x10000000000000ull) << 64; bexp = eb; }
    if (aexp < bexp || (aexp == bexp && ma < mb)) {
        unsigned long long ts = sa; sa = sb; sb = ts;
        int te = aexp; aexp = bexp; bexp = te;
        unsigned __int128 tm = ma; ma = mb; mb = tm;
    }
    int sh = aexp - bexp;
    if (sh >= 128) mb = mb ? 1 : 0;
    else if (sh) {
        unsigned __int128 lost = mb & (((unsigned __int128)1 << sh) - 1);
        mb >>= sh;
        if (lost) mb |= 1;
    }
    if (sa == sb) return cts_dpack((int)sa, aexp, ma + mb, mode);
    if (ma == mb) return mode == 3 ? 0x8000000000000000ull : 0ull;
    return cts_dpack((int)sa, aexp, ma - mb, mode);
}
"""

DARITH_SUB = """
static __device__ unsigned long long cts_dsub(unsigned long long a, unsigned long long b, int mode) {
    if (cts_disnan(a) || cts_disnan(b)) return cts_dadd(a, b, mode);
    return cts_dadd(a, b ^ 0x8000000000000000ull, mode);
}
"""

DARITH_MUL = """
static __device__ unsigned long long cts_dmul(unsigned long long a, unsigned long long b, int mode) {
    if (cts_disnan(a) || cts_disnan(b)) return cts_dquiet(cts_disnan(b) ? b : a);
    int sign = (int)((a >> 63) ^ (b >> 63));
    int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff);
    unsigned long long fa = a & 0xfffffffffffffull, fb = b & 0xfffffffffffffull;
    if (ea == 2047 || eb == 2047) {
        int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
        if ((ea == 2047 && b0) || (eb == 2047 && a0)) return 0xfff8000000000000ull;
        return ((unsigned long long)sign << 63) | 0x7ff0000000000000ull;
    }
    if ((ea | fa) == 0 || (eb | fb) == 0) return sign ? 0x8000000000000000ull : 0ull;
    unsigned __int128 pa = ea ? (fa | 0x10000000000000ull) : fa;
    unsigned __int128 pb = eb ? (fb | 0x10000000000000ull) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 1023;
    unsigned __int128 mag = (pa * pb) << 12;
    return cts_dpack(sign, exp, mag, mode);
}
"""

DARITH_FMA = """
static __device__ unsigned long long cts_dfma(unsigned long long a, unsigned long long b, unsigned long long c, int mode) {
    if (cts_disnan(b)) return cts_dquiet(b);
    if (cts_disnan(c)) return cts_dquiet(c);
    if (cts_disnan(a)) return cts_dquiet(a);
    int signp = (int)((a >> 63) ^ (b >> 63));
    int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff), ec = (int)((c >> 52) & 0x7ff);
    unsigned long long fa = a & 0xfffffffffffffull, fb = b & 0xfffffffffffffull, fc = c & 0xfffffffffffffull;
    int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
    int ainf = ea == 2047, binf = eb == 2047, cinf = ec == 2047;
    unsigned long long sc = c >> 63;
    if ((ainf && b0) || (binf && a0)) return 0xfff8000000000000ull;
    if (ainf || binf) {
        if (cinf && sc != (unsigned long long)signp) return 0xfff8000000000000ull;
        return ((unsigned long long)signp << 63) | 0x7ff0000000000000ull;
    }
    if (cinf) return c;
    if (a0 || b0) return cts_dadd(signp ? 0x8000000000000000ull : 0ull, c, mode);
    unsigned __int128 pa = ea ? (fa | 0x10000000000000ull) : fa;
    unsigned __int128 pb = eb ? (fb | 0x10000000000000ull) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 1023;
    unsigned __int128 mag = (pa * pb) << 12;
    if (mag) {
        while (mag >= ((unsigned __int128)1 << 117)) { mag >>= 1; exp++; }
        while (mag < ((unsigned __int128)1 << 116) && exp > 1) { mag <<= 1; exp--; }
    }
    if ((ec | fc) == 0) return cts_dpack(signp, exp, mag, mode);
    unsigned __int128 mc = ec ? ((unsigned __int128)(fc | 0x10000000000000ull) << 64)
                              : ((unsigned __int128)fc << 64);
    int cexp = ec ? ec : 1;
    int sh = exp - cexp;
    unsigned __int128 acc = mag;
    if (sh >= 0) {
        if (sh >= 128) mc = mc ? 1 : 0;
        else if (sh) {
            unsigned __int128 lost = mc & (((unsigned __int128)1 << sh) - 1);
            mc >>= sh;
            if (lost) mc |= 1;
        }
    } else {
        int shc = -sh;
        if (shc >= 128) acc = acc ? 1 : 0;
        else {
            unsigned __int128 lost = acc & (((unsigned __int128)1 << shc) - 1);
            acc >>= shc;
            if (lost) acc |= 1;
        }
        exp = cexp;
    }
    if ((unsigned long long)signp == sc) return cts_dpack(signp, exp, acc + mc, mode);
    if (acc == mc) return mode == 3 ? 0x8000000000000000ull : 0ull;
    if (acc < mc) return cts_dpack((int)sc, exp, mc - acc, mode);
    return cts_dpack(signp, exp, acc - mc, mode);
}
"""

def _darith_binary(kind: str, mode: str) -> str:
    code = _FARITH_CODE[mode]
    call = f"cts_d{kind}(x.as_bits, y.as_bits, {code})"
    return f"""
static __device__ double cts_d{kind}_{mode}(double a, double b) {{
    union {{ double as_double; unsigned long long as_bits; }} x, y, out;
    x.as_double = a;
    y.as_double = b;
    out.as_bits = {call};
    return out.as_double;
}}
"""


def _darith_fma(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ double cts_dfma_{mode}(double a, double b, double c) {{
    union {{ double as_double; unsigned long long as_bits; }} x, y, z, out;
    x.as_double = a;
    y.as_double = b;
    z.as_double = c;
    out.as_bits = cts_dfma(x.as_bits, y.as_bits, z.as_bits, {code});
    return out.as_double;
}}
"""


DARITH_HELPERS: dict[str, tuple[str, str]] = {}
for _kind in ("add", "sub", "mul"):
    for _mode in ("rn", "rz", "ru", "rd"):
        DARITH_HELPERS[f"__d{_kind}_{_mode}"] = (f"cts_d{_kind}_{_mode}", _darith_binary(_kind, _mode))
for _mode in ("rn", "rz", "ru", "rd"):
    DARITH_HELPERS[f"__fma_{_mode}"] = (f"cts_dfma_{_mode}", _darith_fma(_mode))


def darith_needs_add(names: set[str]) -> bool:
    return any(
        name.startswith("cts_dadd_") or name.startswith("cts_dsub_") or name.startswith("cts_dfma_")
        for name in names
    )


def darith_needs_sub(names: set[str]) -> bool:
    return any(name.startswith("cts_dsub_") for name in names)


def darith_needs_mul(names: set[str]) -> bool:
    return any(name.startswith("cts_dmul_") for name in names)


def darith_needs_fma(names: set[str]) -> bool:
    return any(name.startswith("cts_dfma_") for name in names)


FDIV_ISQRT = """
static __device__ unsigned long long cts_isqrt(unsigned __int128 x, int* sticky) {
    unsigned __int128 rem = 0;
    unsigned long long root = 0;
    for (int i = 63; i >= 0; --i) {
        rem = (rem << 2) | ((x >> (2 * i)) & 3);
        unsigned __int128 trial = ((unsigned __int128)root << 2) | 1;
        if (rem >= trial) {
            rem -= trial;
            root = (root << 1) | 1ull;
        } else {
            root <<= 1;
        }
    }
    *sticky = rem != 0;
    return root;
}
"""

FDIV_ROUND = """
static __device__ unsigned cts_fround(int sign, int exp, unsigned __int128 q, int sticky, int mode) {
    while (q >= ((unsigned __int128)1 << 56)) {
        if (q & 1) sticky = 1;
        q >>= 1;
        exp++;
    }
    while (q && q < ((unsigned __int128)1 << 55) && exp > 1) {
        q <<= 1;
        exp--;
    }
    if (sticky) q |= 1;
    return cts_fpack(sign, exp, (unsigned long long)q, mode);
}
"""

FDIV_DIV = """
static __device__ unsigned cts_fdiv(unsigned a, unsigned b, int mode) {
    unsigned sa = a >> 31, sb = b >> 31;
    int sign = (int)(sa ^ sb);
    int ea = (a >> 23) & 0xff, eb = (b >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu, fb = b & 0x7fffffu;
    if ((ea == 255 && fa) || (eb == 255 && fb)) return 0x7fffffffu;
    int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
    int ainf = ea == 255, binf = eb == 255;
    if ((a0 && b0) || (ainf && binf)) return 0x7fffffffu;
    if (a0 || binf) return sign ? 0x80000000u : 0u;
    if (b0 || ainf) return ((unsigned)sign << 31) | 0x7f800000u;
    unsigned ma = ea ? (fa | 0x800000u) : fa;
    unsigned mb = eb ? (fb | 0x800000u) : fb;
    int exp = (ea ? ea : 1) - (eb ? eb : 1) + 127;
    unsigned __int128 num = (unsigned __int128)ma << 55;
    unsigned __int128 q = num / mb;
    int sticky = (num % mb) != 0;
    return cts_fround(sign, exp, q, sticky, mode);
}
"""

FDIV_SQRT = """
static __device__ unsigned cts_fsqrt(unsigned a, int mode) {
    unsigned sa = a >> 31;
    int ea = (a >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu;
    if (ea == 255) return fa ? 0x7fffffffu : (sa ? 0x7fffffffu : 0x7f800000u);
    if ((ea | fa) == 0) return sa ? 0x80000000u : 0u;
    if (sa) return 0x7fffffffu;
    unsigned M;
    int e;
    if (ea == 0) {
        M = fa;
        int s = 0;
        while ((M & 0x800000u) == 0) { M <<= 1; s++; }
        e = -126 - s;
    } else {
        M = fa | 0x800000u;
        e = ea - 127;
    }
    int half = (e >= 0) ? (e >> 1) : -(((-e) + 1) >> 1);
    unsigned __int128 X = (unsigned __int128)M << ((e & 1) ? 88 : 87);
    int sticky = 0;
    unsigned long long root = cts_isqrt(X, &sticky);
    return cts_fround(0, half + 127, root, sticky, mode);
}
"""

FDIV_RSQRT = """
static __device__ unsigned cts_frsqrt(unsigned a) {
    unsigned sa = a >> 31;
    int ea = (a >> 23) & 0xff;
    unsigned fa = a & 0x7fffffu;
    if (ea == 255) return fa ? 0x7fffffffu : (sa ? 0x7fffffffu : 0u);
    if ((ea | fa) == 0) return sa ? 0xff800000u : 0x7f800000u;
    if (sa) return 0x7fffffffu;
    unsigned M;
    int e;
    if (ea == 0) {
        M = fa;
        int s = 0;
        while ((M & 0x800000u) == 0) { M <<= 1; s++; }
        e = -126 - s;
    } else {
        M = fa | 0x800000u;
        e = ea - 127;
    }
    int half = (e >= 0) ? (e >> 1) : -(((-e) + 1) >> 1);
    unsigned __int128 X = (unsigned __int128)M << ((e & 1) ? 88 : 87);
    int sticky_sqrt = 0;
    unsigned long long root = cts_isqrt(X, &sticky_sqrt);
    unsigned __int128 root2 = (unsigned __int128)root * root;
    unsigned __int128 rem = X - root2;
    unsigned __int128 q = ((unsigned __int128)1 << 110) / root;
    unsigned __int128 qrem = ((unsigned __int128)1 << 110) % root;
    unsigned __int128 corr = 0;
    unsigned __int128 crem = 0;
    if (rem) {
        unsigned __int128 den = root2 << 1;
        unsigned __int128 cnum = q * rem;
        corr = cnum / den;
        crem = cnum % den;
    }
    unsigned __int128 mag = q - corr;
    int sticky = (qrem != 0) || (crem != 0);
    return cts_fround(0, 127 - half, mag, sticky, 0);
}
"""

def _fdiv_binary(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ float cts_fdiv_{mode}(float a, float b) {{
    union {{ float as_float; unsigned as_bits; }} x, y, out;
    x.as_float = a;
    y.as_float = b;
    out.as_bits = cts_fdiv(x.as_bits, y.as_bits, {code});
    return out.as_float;
}}
"""


def _fdiv_rcp(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ float cts_frcp_{mode}(float a) {{
    union {{ float as_float; unsigned as_bits; }} x, out;
    x.as_float = a;
    out.as_bits = cts_fdiv(0x3f800000u, x.as_bits, {code});
    return out.as_float;
}}
"""


def _fdiv_sqrt(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ float cts_fsqrt_{mode}(float a) {{
    union {{ float as_float; unsigned as_bits; }} x, out;
    x.as_float = a;
    out.as_bits = cts_fsqrt(x.as_bits, {code});
    return out.as_float;
}}
"""


FDIV_RSQRT_HELPER = """
static __device__ float cts_frsqrt_rn(float a) {
    union { float as_float; unsigned as_bits; } x, out;
    x.as_float = a;
    out.as_bits = cts_frsqrt(x.as_bits);
    return out.as_float;
}
"""


FDIV_HELPERS: dict[str, tuple[str, str]] = {}
for _mode in ("rn", "rz", "ru", "rd"):
    FDIV_HELPERS[f"__fdiv_{_mode}"] = (f"cts_fdiv_{_mode}", _fdiv_binary(_mode))
    FDIV_HELPERS[f"__frcp_{_mode}"] = (f"cts_frcp_{_mode}", _fdiv_rcp(_mode))
    FDIV_HELPERS[f"__fsqrt_{_mode}"] = (f"cts_fsqrt_{_mode}", _fdiv_sqrt(_mode))
FDIV_HELPERS["__frsqrt_rn"] = ("cts_frsqrt_rn", FDIV_RSQRT_HELPER)


# Directed double division, reciprocal, and square root. NaN payloads are
# kept and quieted; two NaNs keep the second operand. 0/0, inf/inf, and the
# square root of a negative number are the negative quiet NaN.
DDIV_DIV = """
static __device__ unsigned long long cts_ddiv(unsigned long long a, unsigned long long b, int mode) {
    if (cts_disnan(a) || cts_disnan(b)) return cts_dquiet(mode == 0 ? (cts_disnan(b) ? b : a) : (cts_disnan(a) ? a : b));
    int sign = (int)((a >> 63) ^ (b >> 63));
    int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff);
    unsigned long long fa = a & 0xfffffffffffffull, fb = b & 0xfffffffffffffull;
    int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
    int ainf = ea == 2047, binf = eb == 2047;
    if ((a0 && b0) || (ainf && binf)) return 0xfff8000000000000ull;
    if (a0 || binf) return sign ? 0x8000000000000000ull : 0ull;
    if (b0 || ainf) return ((unsigned long long)sign << 63) | 0x7ff0000000000000ull;
    unsigned long long ma = ea ? (fa | 0x10000000000000ull) : fa;
    unsigned long long mb = eb ? (fb | 0x10000000000000ull) : fb;
    int xa = ea ? ea : 1, xb = eb ? eb : 1;
    while (!(ma & 0x10000000000000ull)) { ma <<= 1; xa--; }
    while (!(mb & 0x10000000000000ull)) { mb <<= 1; xb--; }
    unsigned __int128 num = (unsigned __int128)ma << 75;
    unsigned __int128 q = num / mb;
    if (num % mb) q |= 1;
    return cts_dpack(sign, xa - xb + 1023 + 41, q, mode);
}
"""

DDIV_SQRT = """
static __device__ unsigned long long cts_dsqrt(unsigned long long a, int mode) {
    if (cts_disnan(a)) return cts_dquiet(a);
    int ea = (int)((a >> 52) & 0x7ff);
    unsigned long long fa = a & 0xfffffffffffffull;
    if ((ea | fa) == 0) return a;
    if (a >> 63) return 0xfff8000000000000ull;
    if (ea == 2047) return a;
    unsigned long long m = ea ? (fa | 0x10000000000000ull) : fa;
    int e = (ea ? ea : 1) - 1023;
    while (!(m & 0x10000000000000ull)) { m <<= 1; e--; }
    int odd = e & 1;
    int half = (e - odd) / 2;
    unsigned __int128 x = (unsigned __int128)m << (odd ? 75 : 74);
    int sticky = 0;
    unsigned __int128 root = cts_isqrt(x, &sticky);
    if (sticky) root |= 1;
    return cts_dpack(0, half + 1023 + 53, root, mode);
}
"""


def _ddiv_binary(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ double cts_ddiv_{mode}(double a, double b) {{
    union {{ double as_double; unsigned long long as_bits; }} x, y, out;
    x.as_double = a;
    y.as_double = b;
    out.as_bits = cts_ddiv(x.as_bits, y.as_bits, {code});
    return out.as_double;
}}
"""


def _ddiv_rcp(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ double cts_drcp_{mode}(double a) {{
    union {{ double as_double; unsigned long long as_bits; }} x, out;
    x.as_double = a;
    out.as_bits = cts_ddiv(0x3ff0000000000000ull, x.as_bits, {code});
    return out.as_double;
}}
"""


def _ddiv_sqrt(mode: str) -> str:
    code = _FARITH_CODE[mode]
    return f"""
static __device__ double cts_dsqrt_{mode}(double a) {{
    union {{ double as_double; unsigned long long as_bits; }} x, out;
    x.as_double = a;
    out.as_bits = cts_dsqrt(x.as_bits, {code});
    return out.as_double;
}}
"""


DDIV_HELPERS: dict[str, tuple[str, str]] = {}
for _mode in ("rn", "rz", "ru", "rd"):
    DDIV_HELPERS[f"__ddiv_{_mode}"] = (f"cts_ddiv_{_mode}", _ddiv_binary(_mode))
    DDIV_HELPERS[f"__drcp_{_mode}"] = (f"cts_drcp_{_mode}", _ddiv_rcp(_mode))
    DDIV_HELPERS[f"__dsqrt_{_mode}"] = (f"cts_dsqrt_{_mode}", _ddiv_sqrt(_mode))


def ddiv_needs_div(names: set[str]) -> bool:
    return any(name.startswith("cts_ddiv_") or name.startswith("cts_drcp_") for name in names)


def ddiv_needs_sqrt(names: set[str]) -> bool:
    return any(name.startswith("cts_dsqrt_") for name in names)


def fdiv_needs_div(names: set[str]) -> bool:
    return any(name.startswith("cts_fdiv_") or name.startswith("cts_frcp_") for name in names)


def fdiv_needs_sqrt(names: set[str]) -> bool:
    return any(name.startswith("cts_fsqrt_") for name in names)


def fdiv_needs_rsqrt(names: set[str]) -> bool:
    return any(name.startswith("cts_frsqrt_") for name in names)


# Float NaNs become 0x7fffffff. Signed zeros: min is -0 when either is -0,
# max is +0 when either is +0. Two double NaNs passed to min or max keep
# the second operand. nextafter does the same when both operands are NaN,
# and returns the one NaN operand, quieted, when the other is a number.
FMIN_LESS = """
static __device__ int cts_fless(unsigned a, unsigned b) {
    unsigned sa = a >> 31;
    unsigned sb = b >> 31;
    if (sa != sb) return (int)sa;
    if (sa == 0u) return a < b;
    return a > b;
}
"""

FMIN_MINMAX = """
static __device__ unsigned cts_fminmax(unsigned a, unsigned b, int is_max) {
    int na = ((a >> 23) & 0xffu) == 0xffu && (a & 0x7fffffu);
    int nb = ((b >> 23) & 0xffu) == 0xffu && (b & 0x7fffffu);
    if (na && nb) return 0x7fffffffu;
    if (na) return b;
    if (nb) return a;
    if (((a << 1) == 0u) && ((b << 1) == 0u)) {
        if (!is_max) return (a | b) & 0x80000000u;
        return (a & b) & 0x80000000u;
    }
    if (a == b) return a;
    int less = cts_fless(a, b);
    if (is_max) return less ? b : a;
    return less ? a : b;
}
"""

FMIN_NEXT = """
static __device__ unsigned cts_fnext(unsigned x, unsigned y) {
    int xnan = ((x >> 23) & 0xffu) == 0xffu && (x & 0x7fffffu);
    int ynan = ((y >> 23) & 0xffu) == 0xffu && (y & 0x7fffffu);
    if (xnan || ynan) return 0x7fffffffu;
    if (((x << 1) == 0u) && ((y << 1) == 0u)) return y;
    if (x == y) return x;
    if ((x << 1) == 0u) return (y & 0x80000000u) | 1u;
    int up = cts_fless(x, y);
    unsigned neg = x >> 31;
    if (up) return neg ? x - 1u : x + 1u;
    return neg ? x + 1u : x - 1u;
}
"""

DMIN_LESS = """
static __device__ int cts_dless(unsigned long long a, unsigned long long b) {
    unsigned long long sa = a >> 63;
    unsigned long long sb = b >> 63;
    if (sa != sb) return (int)sa;
    if (sa == 0ull) return a < b;
    return a > b;
}
"""

DMIN_MINMAX = """
static __device__ unsigned long long cts_dminmax(unsigned long long a, unsigned long long b, int is_max) {
    int na = ((a >> 52) & 0x7ffull) == 0x7ffull && (a & 0xfffffffffffffull);
    int nb = ((b >> 52) & 0x7ffull) == 0x7ffull && (b & 0xfffffffffffffull);
    if (na && nb) return b | 0x0008000000000000ull;
    if (na) return b;
    if (nb) return a;
    if (((a << 1) == 0ull) && ((b << 1) == 0ull)) {
        if (!is_max) return (a | b) & 0x8000000000000000ull;
        return (a & b) & 0x8000000000000000ull;
    }
    if (a == b) return a;
    int less = cts_dless(a, b);
    if (is_max) return less ? b : a;
    return less ? a : b;
}
"""

DMIN_NEXT = """
static __device__ unsigned long long cts_dnext(unsigned long long x, unsigned long long y) {
    int xnan = ((x >> 52) & 0x7ffull) == 0x7ffull && (x & 0xfffffffffffffull);
    int ynan = ((y >> 52) & 0x7ffull) == 0x7ffull && (y & 0xfffffffffffffull);
    if (ynan) return y | 0x0008000000000000ull;
    if (xnan) return x | 0x0008000000000000ull;
    if (((x << 1) == 0ull) && ((y << 1) == 0ull)) return y;
    if (x == y) return x;
    if ((x << 1) == 0ull) return (y & 0x8000000000000000ull) | 1ull;
    int up = cts_dless(x, y);
    unsigned long long neg = x >> 63;
    if (up) return neg ? x - 1ull : x + 1ull;
    return neg ? x + 1ull : x - 1ull;
}
"""


def _fmin_float(name: str, expr: str) -> str:
    return f"""
static __device__ float {name}(float a, float b) {{
    union {{ float as_float; unsigned as_bits; }} x, y, out;
    x.as_float = a;
    y.as_float = b;
    out.as_bits = {expr};
    return out.as_float;
}}
"""


def _fmin_double(name: str, expr: str) -> str:
    return f"""
static __device__ double {name}(double a, double b) {{
    union {{ double as_double; unsigned long long as_bits; }} x, y, out;
    x.as_double = a;
    y.as_double = b;
    out.as_bits = {expr};
    return out.as_double;
}}
"""


FMIN_HELPERS: dict[str, tuple[str, str]] = {
    "fminf": ("cts_fminf", _fmin_float("cts_fminf", "cts_fminmax(x.as_bits, y.as_bits, 0)")),
    "fmaxf": ("cts_fmaxf", _fmin_float("cts_fmaxf", "cts_fminmax(x.as_bits, y.as_bits, 1)")),
    "fmin": ("cts_fmin", _fmin_double("cts_fmin", "cts_dminmax(x.as_bits, y.as_bits, 0)")),
    "fmax": ("cts_fmax", _fmin_double("cts_fmax", "cts_dminmax(x.as_bits, y.as_bits, 1)")),
    "nextafterf": ("cts_nextafterf", _fmin_float("cts_nextafterf", "cts_fnext(x.as_bits, y.as_bits)")),
    "nextafter": ("cts_nextafter", _fmin_double("cts_nextafter", "cts_dnext(x.as_bits, y.as_bits)")),
    "copysignf": (
        "cts_copysignf",
        _fmin_float("cts_copysignf", "(x.as_bits & 0x7fffffffu) | (y.as_bits & 0x80000000u)"),
    ),
    "copysign": (
        "cts_copysign",
        _fmin_double(
            "cts_copysign",
            "(x.as_bits & 0x7fffffffffffffffull) | (y.as_bits & 0x8000000000000000ull)",
        ),
    ),
}

_FMIN_FLOAT_CMP = {"cts_fminf", "cts_fmaxf"}
_FMIN_FLOAT_NEXT = {"cts_nextafterf"}
_FMIN_DOUBLE_CMP = {"cts_fmin", "cts_fmax"}
_FMIN_DOUBLE_NEXT = {"cts_nextafter"}


def fmin_needs_fless(names: set[str]) -> bool:
    return bool(names & (_FMIN_FLOAT_CMP | _FMIN_FLOAT_NEXT))


def fmin_needs_fminmax(names: set[str]) -> bool:
    return bool(names & _FMIN_FLOAT_CMP)


def fmin_needs_fnext(names: set[str]) -> bool:
    return bool(names & _FMIN_FLOAT_NEXT)


def fmin_needs_dless(names: set[str]) -> bool:
    return bool(names & (_FMIN_DOUBLE_CMP | _FMIN_DOUBLE_NEXT))


def fmin_needs_dminmax(names: set[str]) -> bool:
    return bool(names & _FMIN_DOUBLE_CMP)


def fmin_needs_dnext(names: set[str]) -> bool:
    return bool(names & _FMIN_DOUBLE_NEXT)


# Half keeps 10 explicit bits. A NaN becomes 0x7fff, and widening that NaN
# becomes the float 0x7fffffff. Infinity keeps its sign.
HALF_FROM_FLOAT = """
static __device__ unsigned short cts_hpack(int sign, int exp, unsigned long long mag, int mode) {
    if (mag == 0) return (unsigned short)(sign ? 0x8000 : 0);
    while (mag >= (1ull << 34)) { mag >>= 1; exp++; }
    while (mag < (1ull << 33) && exp > 1) { mag <<= 1; exp--; }
    if (exp < 1 || (exp == 1 && mag < (1ull << 33))) {
        int need = 1 - exp;
        exp = 0;
        if (need >= 64) mag = mag ? 1 : 0;
        else if (need) {
            unsigned long long lost = mag & ((1ull << need) - 1ull);
            mag >>= need;
            if (lost) mag |= 1ull;
        }
    }
    unsigned long long rem = mag & ((1ull << 23) - 1ull);
    unsigned sig = (unsigned)(mag >> 23) & 0x7ffu;
    int toward_inf = sign ? (mode == 3) : (mode == 2);
    int up = 0;
    if (mode == 0) up = rem > (1ull << 22) || (rem == (1ull << 22) && (sig & 1u));
    else if (toward_inf) up = rem != 0;
    if (up) {
        sig += 1u;
        if (exp == 0) {
            if (sig == 0x400u) exp = 1;
        } else if (sig == 0x800u) {
            sig = 0x400u;
            exp++;
        }
    }
    if (exp >= 31) {
        int to_inf = (mode == 0) || (sign ? mode == 3 : mode == 2);
        if (to_inf) return (unsigned short)((sign << 15) | 0x7c00);
        return (unsigned short)((sign << 15) | 0x7bff);
    }
    if (exp <= 0) return (unsigned short)((sign << 15) | (sig & 0x3ffu));
    return (unsigned short)((sign << 15) | (exp << 10) | (sig & 0x3ffu));
}

static __device__ unsigned short cts_f2h(float value, int mode) {
    union { float as_float; unsigned as_bits; } in;
    in.as_float = value;
    unsigned sign = in.as_bits >> 31;
    int exp = (int)((in.as_bits >> 23) & 0xff);
    unsigned frac = in.as_bits & 0x7fffffu;
    if (exp == 255) {
        if (frac) return (unsigned short)0x7fff;
        return (unsigned short)((sign << 15) | 0x7c00);
    }
    unsigned sig = exp ? (frac | 0x800000u) : frac;
    int hexp = exp ? (exp - 112) : -111;
    return cts_hpack((int)sign, hexp, (unsigned long long)sig << 10, mode);
}
"""

HALF_TO_FLOAT = """
static __device__ float cts_h2f(unsigned short bits) {
    unsigned sign = bits >> 15;
    int exp = (bits >> 10) & 0x1f;
    unsigned frac = bits & 0x3ffu;
    union { float as_float; unsigned as_bits; } out;
    if (exp == 31) {
        out.as_bits = frac ? 0x7fffffffu : ((sign << 31) | 0x7f800000u);
        return out.as_float;
    }
    if (exp == 0) {
        out.as_float = (float)frac * 0x1p-24f;
        if (sign) out.as_bits |= 0x80000000u;
        return out.as_float;
    }
    out.as_bits = (sign << 31) | ((unsigned)(exp + 112) << 23) | (frac << 13);
    return out.as_float;
}
"""

HALF_FROM_INT = """
static __device__ unsigned short cts_u2h(unsigned long long mag, int sign, int mode) {
    if (mag == 0) return (unsigned short)(sign ? 0x8000 : 0);
    int k = 0;
    unsigned long long scan = mag;
    while (scan > 1ull) {
        scan >>= 1;
        k++;
    }
    int hexp = k + 15;
    unsigned long long packed;
    if (k <= 33) packed = mag << (33 - k);
    else {
        int need = k - 33;
        unsigned long long lost = mag & ((1ull << need) - 1ull);
        packed = mag >> need;
        if (lost) packed |= 1ull;
    }
    return cts_hpack(sign, hexp, packed, mode);
}

static __device__ unsigned short cts_s2h(long long value, int mode) {
    int sign = value < 0;
    unsigned long long mag = sign ? ((unsigned long long)(-(value + 1)) + 1ull) : (unsigned long long)value;
    return cts_u2h(mag, sign, mode);
}
"""

HALF_FROM_DOUBLE = """
static __device__ unsigned short cts_d2h(double value) {
    union { double as_double; unsigned long long as_bits; } in;
    in.as_double = value;
    unsigned sign = (unsigned)(in.as_bits >> 63);
    int exp = (int)((in.as_bits >> 52) & 0x7ff);
    unsigned long long frac = in.as_bits & 0xfffffffffffffull;
    if (exp == 2047) {
        if (frac) return (unsigned short)0x7fff;
        return (unsigned short)((sign << 15) | 0x7c00);
    }
    unsigned long long sig = exp ? (frac | (1ull << 52)) : frac;
    int hexp = exp ? (exp - 1008) : (1 - 1008);
    unsigned long long lost = sig & ((1ull << 19) - 1ull);
    unsigned long long mag = sig >> 19;
    if (lost) mag |= 1ull;
    return cts_hpack((int)sign, hexp, mag, 0);
}
"""

# Every finite half is exact in float. Short saturates past 32767.
# A NaN is 0 for 16-bit and 32-bit results, and the high bit for 64-bit.
HALF_TO_INT = """
static __device__ unsigned long long cts_h2i(unsigned short bits, int mode, int kind) {
    float x = cts_h2f(bits);
    float y = rintf(x);
    if (mode == 1) y = truncf(x);
    else if (mode == 2) y = ceilf(x);
    else if (mode == 3) y = floorf(x);
    int nan = !(y == y);
    if (kind == 0) {
        if (nan) return 0ull;
        if (y >= 2147483648.0f) return 2147483647ull;
        if (y < -2147483648.0f) return 0x80000000ull;
        return (unsigned)(int)y;
    }
    if (kind == 1) {
        if (nan || y <= 0.0f) return 0ull;
        if (y >= 4294967296.0f) return 0xffffffffull;
        return (unsigned)y;
    }
    if (kind == 2) {
        if (nan) return 0ull;
        if (y >= 32768.0f) return 32767ull;
        if (y < -32768.0f) return 0x8000ull;
        return (unsigned short)(short)(int)y;
    }
    if (kind == 3) {
        if (nan || y <= 0.0f) return 0ull;
        if (y >= 65536.0f) return 65535ull;
        return (unsigned short)y;
    }
    if (kind == 4) {
        if (nan) return 0x8000000000000000ull;
        if (y >= 0x1p63f) return 0x7fffffffffffffffull;
        if (y <= -0x1p63f) return 0x8000000000000000ull;
        return (unsigned long long)(long long)y;
    }
    if (nan) return 0x8000000000000000ull;
    if (y <= 0.0f) return 0ull;
    if (y >= 0x1p64f) return 0xffffffffffffffffull;
    return (unsigned long long)y;
}
"""

_HALF_CODE = {"rn": 0, "rz": 1, "ru": 2, "rd": 3}


def _float_to_half(mode: str) -> str:
    name = f"cts_float2half_{mode}"
    code = _HALF_CODE[mode]
    return f"""
static __device__ __half {name}(float x) {{
    union {{ unsigned short as_bits; __half as_half; }} out;
    out.as_bits = cts_f2h(x, {code});
    return out.as_half;
}}
"""


HALF_HELPERS: dict[str, tuple[str, str]] = {}
for _mode in ("rn", "rz", "ru", "rd"):
    HALF_HELPERS[f"__float2half_{_mode}"] = (f"cts_float2half_{_mode}", _float_to_half(_mode))
HALF_HELPERS["__float2half"] = HALF_HELPERS["__float2half_rn"]


def _int_to_half(ctype: str, stem: str, mode: str, signed: bool) -> str:
    code = _HALF_CODE[mode]
    if signed:
        call = f"cts_s2h((long long)x, {code})"
    else:
        call = f"cts_u2h((unsigned long long)x, 0, {code})"
    return f"""
static __device__ __half {stem}_{mode}({ctype} x) {{
    union {{ unsigned short as_bits; __half as_half; }} out;
    out.as_bits = {call};
    return out.as_half;
}}
"""


for _ctype, _stem, _signed in (
    ("int", "cts_int2half", True),
    ("unsigned int", "cts_uint2half", False),
    ("short", "cts_short2half", True),
    ("unsigned short", "cts_ushort2half", False),
    ("long long", "cts_ll2half", True),
    ("unsigned long long", "cts_ull2half", False),
):
    for _mode in ("rn", "rz", "ru", "rd"):
        HALF_HELPERS[f"__{_stem[4:]}_{_mode}"] = (
            f"{_stem}_{_mode}",
            _int_to_half(_ctype, _stem, _mode, _signed),
        )

HALF_HELPERS["__double2half"] = (
    "cts_double2half",
    """
static __device__ __half cts_double2half(double x) {
    union { unsigned short as_bits; __half as_half; } out;
    out.as_bits = cts_d2h(x);
    return out.as_half;
}
""",
)
def _half_out(ret: str, stem: str, mode: str, kind: int, pun: str) -> str:
    code = _HALF_CODE[mode]
    if pun == "u64":
        return f"""
static __device__ unsigned long long {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    in.as_half = value;
    return cts_h2i(in.as_bits, {code}, {kind});
}}
"""
    if pun == "u32":
        return f"""
static __device__ unsigned {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    in.as_half = value;
    return (unsigned)cts_h2i(in.as_bits, {code}, {kind});
}}
"""
    if pun == "u16":
        return f"""
static __device__ unsigned short {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    in.as_half = value;
    return (unsigned short)cts_h2i(in.as_bits, {code}, {kind});
}}
"""
    if pun == "i16":
        return f"""
static __device__ short {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    union {{ unsigned short as_bits; short as_value; }} out;
    in.as_half = value;
    out.as_bits = (unsigned short)cts_h2i(in.as_bits, {code}, {kind});
    return out.as_value;
}}
"""
    if pun == "i64":
        return f"""
static __device__ long long {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    union {{ unsigned long long as_bits; long long as_value; }} out;
    in.as_half = value;
    out.as_bits = cts_h2i(in.as_bits, {code}, {kind});
    return out.as_value;
}}
"""
    return f"""
static __device__ int {stem}_{mode}(__half value) {{
    union {{ unsigned short as_bits; __half as_half; }} in;
    union {{ unsigned as_bits; int as_value; }} out;
    in.as_half = value;
    out.as_bits = (unsigned)cts_h2i(in.as_bits, {code}, {kind});
    return out.as_value;
}}
"""


for _ret, _stem, _kind, _pun in (
    ("int", "cts_half2int", 0, "i32"),
    ("unsigned", "cts_half2uint", 1, "u32"),
    ("short", "cts_half2short", 2, "i16"),
    ("unsigned short", "cts_half2ushort", 3, "u16"),
    ("long long", "cts_half2ll", 4, "i64"),
    ("unsigned long long", "cts_half2ull", 5, "u64"),
):
    for _mode in ("rn", "rz", "ru", "rd"):
        HALF_HELPERS[f"__{_stem[4:]}_{_mode}"] = (
            f"{_stem}_{_mode}",
            _half_out(_ret, _stem, _mode, _kind, _pun),
        )

HALF_HELPERS["__half2float"] = (
    "cts_half2float",
    """
static __device__ float cts_half2float(__half value) {
    union { unsigned short as_bits; __half as_half; } in;
    in.as_half = value;
    return cts_h2f(in.as_bits);
}
""",
)


def half_needs_from_float(names: set[str]) -> bool:
    prefixes = (
        "cts_float2half",
        "cts_int2half_",
        "cts_uint2half_",
        "cts_short2half_",
        "cts_ushort2half_",
        "cts_ll2half_",
        "cts_ull2half_",
        "cts_double2half",
    )
    return any(name.startswith(prefixes) for name in names)


def half_needs_int(names: set[str]) -> bool:
    prefixes = (
        "cts_int2half_",
        "cts_uint2half_",
        "cts_short2half_",
        "cts_ushort2half_",
        "cts_ll2half_",
        "cts_ull2half_",
    )
    return any(name.startswith(prefixes) for name in names)


def half_needs_double(names: set[str]) -> bool:
    return any(name.startswith("cts_double2half") for name in names)


def half_needs_to_int(names: set[str]) -> bool:
    prefixes = (
        "cts_half2int_",
        "cts_half2uint_",
        "cts_half2short_",
        "cts_half2ushort_",
        "cts_half2ll_",
        "cts_half2ull_",
    )
    return any(name.startswith(prefixes) for name in names)


def half_needs_to_float(names: set[str]) -> bool:
    return any(name.startswith("cts_half2float") for name in names) or half_needs_to_int(names)



# Half arithmetic is nearest-even only. A NaN is 0x7fff. Exact cancellation
# of opposite signs is +0. The saturate forms clamp into [0, 1].
HARITH_ADD = """
static __device__ unsigned short cts_f16add(unsigned short a, unsigned short b) {
    unsigned sa = a >> 15, sb = b >> 15;
    int ea = (a >> 10) & 0x1f, eb = (b >> 10) & 0x1f;
    unsigned fa = a & 0x3ffu, fb = b & 0x3ffu;
    if (ea == 31 || eb == 31) {
        if ((ea == 31 && fa) || (eb == 31 && fb)) return (unsigned short)0x7fff;
        if (ea == 31 && eb == 31) return sa == sb ? (unsigned short)((sa << 15) | 0x7c00) : (unsigned short)0x7fff;
        return ea == 31 ? (unsigned short)((sa << 15) | 0x7c00) : (unsigned short)((sb << 15) | 0x7c00);
    }
    unsigned long long ma, mb;
    int aexp, bexp;
    if (ea == 0) { ma = (unsigned long long)fa << 23; aexp = 1; }
    else { ma = (unsigned long long)(fa | 0x400u) << 23; aexp = ea; }
    if (eb == 0) { mb = (unsigned long long)fb << 23; bexp = 1; }
    else { mb = (unsigned long long)(fb | 0x400u) << 23; bexp = eb; }
    if (aexp < bexp || (aexp == bexp && ma < mb)) {
        unsigned ts = sa; sa = sb; sb = ts;
        int te = aexp; aexp = bexp; bexp = te;
        unsigned long long tm = ma; ma = mb; mb = tm;
    }
    int sh = aexp - bexp;
    if (sh >= 63) mb = mb ? 1ull : 0ull;
    else if (sh) {
        unsigned long long lost = mb & ((1ull << sh) - 1ull);
        mb >>= sh;
        if (lost) mb |= 1ull;
    }
    if (sa == sb) return cts_hpack((int)sa, aexp, ma + mb, 0);
    if (ma == mb) return 0;
    return cts_hpack((int)sa, aexp, ma - mb, 0);
}
"""

HARITH_MUL = """
static __device__ unsigned short cts_f16mul(unsigned short a, unsigned short b) {
    unsigned sa = a >> 15, sb = b >> 15;
    int sign = (int)(sa ^ sb);
    int ea = (a >> 10) & 0x1f, eb = (b >> 10) & 0x1f;
    unsigned fa = a & 0x3ffu, fb = b & 0x3ffu;
    if (ea == 31 || eb == 31) {
        if ((ea == 31 && fa) || (eb == 31 && fb)) return (unsigned short)0x7fff;
        int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
        if ((ea == 31 && b0) || (eb == 31 && a0)) return (unsigned short)0x7fff;
        return (unsigned short)((sign << 15) | 0x7c00);
    }
    if ((ea | fa) == 0 || (eb | fb) == 0) return (unsigned short)(sign ? 0x8000 : 0);
    unsigned long long ma = ea ? (fa | 0x400u) : fa;
    unsigned long long mb = eb ? (fb | 0x400u) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 15;
    unsigned long long mag = (ma * mb) << 13;
    return cts_hpack(sign, exp, mag, 0);
}
"""

HARITH_FMA = """
static __device__ unsigned short cts_f16fma(unsigned short a, unsigned short b, unsigned short c) {
    unsigned sa = a >> 15, sb = b >> 15, sc = c >> 15;
    int signp = (int)(sa ^ sb);
    int ea = (a >> 10) & 0x1f, eb = (b >> 10) & 0x1f, ec = (c >> 10) & 0x1f;
    unsigned fa = a & 0x3ffu, fb = b & 0x3ffu, fc = c & 0x3ffu;
    if ((ea == 31 && fa) || (eb == 31 && fb) || (ec == 31 && fc)) return (unsigned short)0x7fff;
    int a0 = (ea | fa) == 0, b0 = (eb | fb) == 0;
    int ainf = ea == 31, binf = eb == 31, cinf = ec == 31;
    if ((ainf && b0) || (binf && a0)) return (unsigned short)0x7fff;
    if (ainf || binf) {
        if (cinf && sc != (unsigned)signp) return (unsigned short)0x7fff;
        return (unsigned short)((signp << 15) | 0x7c00);
    }
    if (a0 || b0) return cts_f16add((unsigned short)(signp ? 0x8000 : 0), c);
    unsigned long long pa = ea ? (fa | 0x400u) : fa;
    unsigned long long pb = eb ? (fb | 0x400u) : fb;
    int exp = (ea ? ea : 1) + (eb ? eb : 1) - 15;
    unsigned long long mag = (pa * pb) << 13;
    if (mag) {
        while (mag >= (1ull << 34)) { mag >>= 1; exp++; }
        while (mag < (1ull << 33) && exp > 1) { mag <<= 1; exp--; }
    }
    if ((ec | fc) == 0) return cts_hpack(signp, exp, mag, 0);
    unsigned long long mc = ec ? ((unsigned long long)(fc | 0x400u) << 23) : ((unsigned long long)fc << 23);
    int cexp = ec ? ec : 1;
    int sh = exp - cexp;
    unsigned long long acc = mag;
    if (sh >= 0) {
        if (sh >= 63) mc = mc ? 1ull : 0ull;
        else if (sh) {
            unsigned long long lost = mc & ((1ull << sh) - 1ull);
            mc >>= sh;
            if (lost) mc |= 1ull;
        }
    } else {
        int shc = -sh;
        if (shc >= 63) acc = acc ? 1ull : 0ull;
        else {
            unsigned long long lost = acc & ((1ull << shc) - 1ull);
            acc >>= shc;
            if (lost) acc |= 1ull;
        }
        exp = cexp;
    }
    if ((unsigned)signp == sc) return cts_hpack(signp, exp, acc + mc, 0);
    if (acc == mc) return 0;
    if (acc < mc) return cts_hpack((int)sc, exp, mc - acc, 0);
    return cts_hpack(signp, exp, acc - mc, 0);
}
"""

HARITH_SAT = """
static __device__ unsigned short cts_f16sat(unsigned short bits) {
    unsigned sign = bits >> 15;
    unsigned exp = (bits >> 10) & 0x1f;
    unsigned frac = bits & 0x3ffu;
    if ((exp == 31 && frac) || sign) return 0;
    if (exp >= 15) return (unsigned short)0x3c00;
    return bits;
}
"""


def _harith_binary(stem: str, call: str) -> str:
    return f"""
static __device__ __half {stem}(__half a, __half b) {{
    union {{ unsigned short as_bits; __half as_half; }} x, y, out;
    x.as_half = a;
    y.as_half = b;
    out.as_bits = {call};
    return out.as_half;
}}
"""


def _harith_fma(stem: str, sat: bool) -> str:
    call = "cts_f16fma(x.as_bits, y.as_bits, z.as_bits)"
    if sat:
        call = f"cts_f16sat({call})"
    return f"""
static __device__ __half {stem}(__half a, __half b, __half c) {{
    union {{ unsigned short as_bits; __half as_half; }} x, y, z, out;
    x.as_half = a;
    y.as_half = b;
    z.as_half = c;
    out.as_bits = {call};
    return out.as_half;
}}
"""


HARITH_HELPERS: dict[str, tuple[str, str]] = {
    "__hadd": ("cts_f16add", _harith_binary("cts_f16add", "cts_f16add(x.as_bits, y.as_bits)")),
    "__hsub": (
        "cts_f16sub",
        _harith_binary("cts_f16sub", "cts_f16add(x.as_bits, (unsigned short)(y.as_bits ^ 0x8000u))"),
    ),
    "__hmul": ("cts_f16mul", _harith_binary("cts_f16mul", "cts_f16mul(x.as_bits, y.as_bits)")),
    "__hadd_sat": (
        "cts_f16add_sat",
        _harith_binary("cts_f16add_sat", "cts_f16sat(cts_f16add(x.as_bits, y.as_bits))"),
    ),
    "__hsub_sat": (
        "cts_f16sub_sat",
        _harith_binary("cts_f16sub_sat", "cts_f16sat(cts_f16add(x.as_bits, (unsigned short)(y.as_bits ^ 0x8000u)))"),
    ),
    "__hmul_sat": (
        "cts_f16mul_sat",
        _harith_binary("cts_f16mul_sat", "cts_f16sat(cts_f16mul(x.as_bits, y.as_bits))"),
    ),
    "__hfma": ("cts_f16fma", _harith_fma("cts_f16fma", False)),
    "__hfma_sat": ("cts_f16fma_sat", _harith_fma("cts_f16fma_sat", True)),
}
HARITH_HELPERS["__hadd_rn"] = HARITH_HELPERS["__hadd"]
HARITH_HELPERS["__hsub_rn"] = HARITH_HELPERS["__hsub"]
HARITH_HELPERS["__hmul_rn"] = HARITH_HELPERS["__hmul"]


def _scalar_half_name(name: str, stem: str) -> bool:
    return name.startswith(stem) and not name.startswith(stem + "2")


def harith_needs_add(names: set[str]) -> bool:
    return any(
        _scalar_half_name(name, "cts_f16add")
        or _scalar_half_name(name, "cts_f16sub")
        or _scalar_half_name(name, "cts_f16fma")
        for name in names
    )


def harith_needs_mul(names: set[str]) -> bool:
    return any(_scalar_half_name(name, "cts_f16mul") for name in names)


def harith_needs_fma(names: set[str]) -> bool:
    return any(_scalar_half_name(name, "cts_f16fma") for name in names)


def harith_needs_sat(names: set[str]) -> bool:
    return any(name.endswith("_sat") for name in names)


# Packed half2 is two independent lanes. The low lane occupies bits 0..15.
H2_ADD = """
static __device__ unsigned cts_f16add2(unsigned a, unsigned b) {
    unsigned lo = cts_f16add((unsigned short)a, (unsigned short)b);
    unsigned hi = cts_f16add((unsigned short)(a >> 16), (unsigned short)(b >> 16));
    return lo | (hi << 16);
}
"""

H2_MUL = """
static __device__ unsigned cts_f16mul2(unsigned a, unsigned b) {
    unsigned lo = cts_f16mul((unsigned short)a, (unsigned short)b);
    unsigned hi = cts_f16mul((unsigned short)(a >> 16), (unsigned short)(b >> 16));
    return lo | (hi << 16);
}
"""

H2_FMA = """
static __device__ unsigned cts_f16fma2(unsigned a, unsigned b, unsigned c) {
    unsigned lo = cts_f16fma((unsigned short)a, (unsigned short)b, (unsigned short)c);
    unsigned hi = cts_f16fma((unsigned short)(a >> 16), (unsigned short)(b >> 16), (unsigned short)(c >> 16));
    return lo | (hi << 16);
}
"""

H2_SAT = """
static __device__ unsigned cts_f16sat2(unsigned bits) {
    unsigned lo = cts_f16sat((unsigned short)bits);
    unsigned hi = cts_f16sat((unsigned short)(bits >> 16));
    return lo | (hi << 16);
}
"""


def _h2_binary(stem: str, call: str) -> str:
    return f"""
static __device__ __half2 {stem}(__half2 a, __half2 b) {{
    union {{ __half2 as_h; unsigned as_bits; }} x, y, out;
    x.as_h = a;
    y.as_h = b;
    out.as_bits = {call};
    return out.as_h;
}}
"""


def _h2_ternary(stem: str, call: str) -> str:
    return f"""
static __device__ __half2 {stem}(__half2 a, __half2 b, __half2 c) {{
    union {{ __half2 as_h; unsigned as_bits; }} x, y, z, out;
    x.as_h = a;
    y.as_h = b;
    z.as_h = c;
    out.as_bits = {call};
    return out.as_h;
}}
"""


H2_HELPERS: dict[str, tuple[str, str]] = {
    "__hadd2": ("cts_f16add2", _h2_binary("cts_f16add2", "cts_f16add2(x.as_bits, y.as_bits)")),
    "__hsub2": ("cts_f16sub2", _h2_binary("cts_f16sub2", "cts_f16add2(x.as_bits, y.as_bits ^ 0x80008000u)")),
    "__hmul2": ("cts_f16mul2", _h2_binary("cts_f16mul2", "cts_f16mul2(x.as_bits, y.as_bits)")),
    "__hadd2_sat": (
        "cts_f16add2_sat",
        _h2_binary("cts_f16add2_sat", "cts_f16sat2(cts_f16add2(x.as_bits, y.as_bits))"),
    ),
    "__hsub2_sat": (
        "cts_f16sub2_sat",
        _h2_binary("cts_f16sub2_sat", "cts_f16sat2(cts_f16add2(x.as_bits, y.as_bits ^ 0x80008000u))"),
    ),
    "__hmul2_sat": (
        "cts_f16mul2_sat",
        _h2_binary("cts_f16mul2_sat", "cts_f16sat2(cts_f16mul2(x.as_bits, y.as_bits))"),
    ),
    "__hfma2": ("cts_f16fma2", _h2_ternary("cts_f16fma2", "cts_f16fma2(x.as_bits, y.as_bits, z.as_bits)")),
    "__hfma2_sat": (
        "cts_f16fma2_sat",
        _h2_ternary("cts_f16fma2_sat", "cts_f16sat2(cts_f16fma2(x.as_bits, y.as_bits, z.as_bits))"),
    ),
    "__float2half2_rn": (
        "cts_float2half2_rn",
        """
static __device__ __half2 cts_float2half2_rn(float a) {
    union { __half2 as_h; unsigned as_bits; } out;
    unsigned bits = cts_f2h(a, 0);
    out.as_bits = bits | (bits << 16);
    return out.as_h;
}
""",
    ),
    "__floats2half2_rn": (
        "cts_floats2half2_rn",
        """
static __device__ __half2 cts_floats2half2_rn(float a, float b) {
    union { __half2 as_h; unsigned as_bits; } out;
    out.as_bits = cts_f2h(a, 0) | (cts_f2h(b, 0) << 16);
    return out.as_h;
}
""",
    ),
    "__float22half2_rn": (
        "cts_float22half2_rn",
        """
static __device__ __half2 cts_float22half2_rn(float2 a) {
    union { __half2 as_h; unsigned as_bits; } out;
    out.as_bits = cts_f2h(a.x, 0) | (cts_f2h(a.y, 0) << 16);
    return out.as_h;
}
""",
    ),
    "__half22float2": (
        "cts_half22float2",
        """
static __device__ float2 cts_half22float2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in;
    in.as_h = a;
    return make_float2(cts_h2f((unsigned short)in.as_bits), cts_h2f((unsigned short)(in.as_bits >> 16)));
}
""",
    ),
    "__halves2half2": (
        "cts_halves2half2",
        """
static __device__ __half2 cts_halves2half2(__half a, __half b) {
    union { unsigned short as_bits; __half as_half; } lo, hi;
    union { __half2 as_h; unsigned as_bits; } out;
    lo.as_half = a;
    hi.as_half = b;
    out.as_bits = (unsigned)lo.as_bits | ((unsigned)hi.as_bits << 16);
    return out.as_h;
}
""",
    ),
    "__low2half": (
        "cts_low2half",
        """
static __device__ __half cts_low2half(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in;
    union { unsigned short as_bits; __half as_half; } lo;
    in.as_h = a;
    lo.as_bits = (unsigned short)in.as_bits;
    return lo.as_half;
}
""",
    ),
    "__high2half": (
        "cts_high2half",
        """
static __device__ __half cts_high2half(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in;
    union { unsigned short as_bits; __half as_half; } hi;
    in.as_h = a;
    hi.as_bits = (unsigned short)(in.as_bits >> 16);
    return hi.as_half;
}
""",
    ),
    "__low2half2": (
        "cts_low2half2",
        """
static __device__ __half2 cts_low2half2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in, out;
    in.as_h = a;
    unsigned lo = in.as_bits & 0xffffu;
    out.as_bits = lo | (lo << 16);
    return out.as_h;
}
""",
    ),
    "__high2half2": (
        "cts_high2half2",
        """
static __device__ __half2 cts_high2half2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in, out;
    in.as_h = a;
    unsigned hi = in.as_bits >> 16;
    out.as_bits = hi | (hi << 16);
    return out.as_h;
}
""",
    ),
    "__lows2half2": (
        "cts_lows2half2",
        """
static __device__ __half2 cts_lows2half2(__half2 a, __half2 b) {
    union { __half2 as_h; unsigned as_bits; } x, y, out;
    x.as_h = a;
    y.as_h = b;
    out.as_bits = (x.as_bits & 0xffffu) | ((y.as_bits & 0xffffu) << 16);
    return out.as_h;
}
""",
    ),
    "__highs2half2": (
        "cts_highs2half2",
        """
static __device__ __half2 cts_highs2half2(__half2 a, __half2 b) {
    union { __half2 as_h; unsigned as_bits; } x, y, out;
    x.as_h = a;
    y.as_h = b;
    out.as_bits = (x.as_bits >> 16) | ((y.as_bits >> 16) << 16);
    return out.as_h;
}
""",
    ),
}
H2_HELPERS["__hadd2_rn"] = H2_HELPERS["__hadd2"]
H2_HELPERS["__hsub2_rn"] = H2_HELPERS["__hsub2"]
H2_HELPERS["__hmul2_rn"] = H2_HELPERS["__hmul2"]


def h2_arity(spelling: str) -> int:
    if spelling.startswith("__hfma2"):
        return 3
    if spelling in {
        "__float2half2_rn",
        "__float22half2_rn",
        "__half22float2",
        "__low2half",
        "__high2half",
        "__low2half2",
        "__high2half2",
    }:
        return 1
    return 2


def h2_needs_add(names: set[str]) -> bool:
    return any(name.startswith(("cts_f16add2", "cts_f16sub2")) for name in names)


def h2_needs_mul(names: set[str]) -> bool:
    return any(name.startswith("cts_f16mul2") for name in names)


def h2_needs_fma(names: set[str]) -> bool:
    return any(name.startswith("cts_f16fma2") for name in names)


def h2_needs_sat(names: set[str]) -> bool:
    return any(name.endswith("_sat") for name in names)


def h2_needs_from_float(names: set[str]) -> bool:
    return any(name.startswith(("cts_float2half2_", "cts_floats2half2_", "cts_float22half2_")) for name in names)


def h2_needs_to_float(names: set[str]) -> bool:
    return any(name.startswith("cts_half22float2") for name in names)


# Ordered compares are false when either value is NaN. Unordered compares
# are true. A half2 predicate lane is 0x3c00 or 0. A mask lane is 0xffff or 0.
HPRED_NAN = """
static __device__ int cts_hnan(unsigned short bits) {
    return ((bits >> 10) & 31) == 31 && (bits & 0x3ffu);
}
"""

HPRED_SIGN = """
static __device__ unsigned short cts_hsign(unsigned short bits, int neg) {
    if (cts_hnan(bits)) return (unsigned short)0x7fff;
    if (neg) return (unsigned short)(bits ^ 0x8000u);
    return (unsigned short)(bits & 0x7fffu);
}
"""

HPRED_CMP = """
static __device__ int cts_hcmp(unsigned short a, unsigned short b, int op) {
    if (cts_hnan(a) || cts_hnan(b)) return (op & 8) ? 1 : 0;
    float fa = cts_h2f(a), fb = cts_h2f(b);
    int base = op & 7;
    if (base == 0) return fa == fb;
    if (base == 1) return fa != fb;
    if (base == 2) return fa < fb;
    if (base == 3) return fa <= fb;
    if (base == 4) return fa > fb;
    return fa >= fb;
}

static __device__ unsigned cts_hcmp2(unsigned a, unsigned b, int op) {
    unsigned lo = cts_hcmp((unsigned short)a, (unsigned short)b, op) ? 0x3c00u : 0u;
    unsigned hi = cts_hcmp((unsigned short)(a >> 16), (unsigned short)(b >> 16), op) ? 0x3c00u : 0u;
    return lo | (hi << 16);
}

static __device__ unsigned cts_hcmp2_mask(unsigned a, unsigned b, int op) {
    unsigned lo = cts_hcmp((unsigned short)a, (unsigned short)b, op) ? 0xffffu : 0u;
    unsigned hi = cts_hcmp((unsigned short)(a >> 16), (unsigned short)(b >> 16), op) ? 0xffffu : 0u;
    return lo | (hi << 16);
}
"""

HPRED_MINMAX = """
static __device__ unsigned short cts_hminmax(unsigned short a, unsigned short b, int ismin) {
    int na = cts_hnan(a), nb = cts_hnan(b);
    if (na && nb) return (unsigned short)0x7fff;
    if (na) return b;
    if (nb) return a;
    if ((a & 0x7fffu) == 0 && (b & 0x7fffu) == 0) {
        if (a == b) return a;
        return (unsigned short)(ismin ? 0x8000 : 0);
    }
    float fa = cts_h2f(a), fb = cts_h2f(b);
    if (ismin) return fa <= fb ? a : b;
    return fa >= fb ? a : b;
}
"""


def _hpred_bool(name: str, code: int) -> str:
    return f"""
static __device__ bool {name}(__half a, __half b) {{
    union {{ unsigned short as_bits; __half as_half; }} x, y;
    x.as_half = a;
    y.as_half = b;
    return cts_hcmp(x.as_bits, y.as_bits, {code});
}}
"""


def _hpred_pair(name: str, code: int) -> str:
    return f"""
static __device__ __half2 {name}(__half2 a, __half2 b) {{
    union {{ __half2 as_h; unsigned as_bits; }} x, y, out;
    x.as_h = a;
    y.as_h = b;
    out.as_bits = cts_hcmp2(x.as_bits, y.as_bits, {code});
    return out.as_h;
}}
"""


def _hpred_mask(name: str, code: int) -> str:
    return f"""
static __device__ unsigned {name}(__half2 a, __half2 b) {{
    union {{ __half2 as_h; unsigned as_bits; }} x, y;
    x.as_h = a;
    y.as_h = b;
    return cts_hcmp2_mask(x.as_bits, y.as_bits, {code});
}}
"""


def _hpred_both(name: str, code: int) -> str:
    return f"""
static __device__ bool {name}(__half2 a, __half2 b) {{
    union {{ __half2 as_h; unsigned as_bits; }} x, y;
    x.as_h = a;
    y.as_h = b;
    return cts_hcmp2(x.as_bits, y.as_bits, {code}) == 0x3c003c00u;
}}
"""


_HPRED_OPS = (
    ("eq", 0),
    ("ne", 1),
    ("lt", 2),
    ("le", 3),
    ("gt", 4),
    ("ge", 5),
    ("equ", 8),
    ("neu", 9),
    ("ltu", 10),
    ("leu", 11),
    ("gtu", 12),
    ("geu", 13),
)

HPRED_HELPERS: dict[str, tuple[str, str]] = {}
for _op, _code in _HPRED_OPS:
    HPRED_HELPERS[f"__h{_op}"] = (f"cts_h{_op}", _hpred_bool(f"cts_h{_op}", _code))
    HPRED_HELPERS[f"__h{_op}2"] = (f"cts_h{_op}2", _hpred_pair(f"cts_h{_op}2", _code))
    HPRED_HELPERS[f"__h{_op}2_mask"] = (f"cts_h{_op}2_mask", _hpred_mask(f"cts_h{_op}2_mask", _code))
    HPRED_HELPERS[f"__hb{_op}2"] = (f"cts_hb{_op}2", _hpred_both(f"cts_hb{_op}2", _code))

HPRED_HELPERS["__hisnan"] = (
    "cts_hisnan",
    """
static __device__ bool cts_hisnan(__half a) {
    union { unsigned short as_bits; __half as_half; } x;
    x.as_half = a;
    return cts_hnan(x.as_bits);
}
""",
)
HPRED_HELPERS["__hisnan2"] = (
    "cts_hisnan2",
    """
static __device__ __half2 cts_hisnan2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in, out;
    in.as_h = a;
    unsigned lo = cts_hnan((unsigned short)in.as_bits) ? 0x3c00u : 0u;
    unsigned hi = cts_hnan((unsigned short)(in.as_bits >> 16)) ? 0x3c00u : 0u;
    out.as_bits = lo | (hi << 16);
    return out.as_h;
}
""",
)
HPRED_HELPERS["__hneg"] = (
    "cts_hneg",
    """
static __device__ __half cts_hneg(__half a) {
    union { unsigned short as_bits; __half as_half; } x, out;
    x.as_half = a;
    out.as_bits = cts_hsign(x.as_bits, 1);
    return out.as_half;
}
""",
)
HPRED_HELPERS["__habs"] = (
    "cts_habs",
    """
static __device__ __half cts_habs(__half a) {
    union { unsigned short as_bits; __half as_half; } x, out;
    x.as_half = a;
    out.as_bits = cts_hsign(x.as_bits, 0);
    return out.as_half;
}
""",
)
HPRED_HELPERS["__hneg2"] = (
    "cts_hneg2",
    """
static __device__ __half2 cts_hneg2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in, out;
    in.as_h = a;
    unsigned lo = cts_hsign((unsigned short)in.as_bits, 1);
    unsigned hi = cts_hsign((unsigned short)(in.as_bits >> 16), 1);
    out.as_bits = lo | (hi << 16);
    return out.as_h;
}
""",
)
HPRED_HELPERS["__habs2"] = (
    "cts_habs2",
    """
static __device__ __half2 cts_habs2(__half2 a) {
    union { __half2 as_h; unsigned as_bits; } in, out;
    in.as_h = a;
    unsigned lo = cts_hsign((unsigned short)in.as_bits, 0);
    unsigned hi = cts_hsign((unsigned short)(in.as_bits >> 16), 0);
    out.as_bits = lo | (hi << 16);
    return out.as_h;
}
""",
)
HPRED_HELPERS["__hmax"] = (
    "cts_hmax",
    """
static __device__ __half cts_hmax(__half a, __half b) {
    union { unsigned short as_bits; __half as_half; } x, y, out;
    x.as_half = a;
    y.as_half = b;
    out.as_bits = cts_hminmax(x.as_bits, y.as_bits, 0);
    return out.as_half;
}
""",
)
HPRED_HELPERS["__hmin"] = (
    "cts_hmin",
    """
static __device__ __half cts_hmin(__half a, __half b) {
    union { unsigned short as_bits; __half as_half; } x, y, out;
    x.as_half = a;
    y.as_half = b;
    out.as_bits = cts_hminmax(x.as_bits, y.as_bits, 1);
    return out.as_half;
}
""",
)
HPRED_HELPERS["__hmax2"] = (
    "cts_hmax2",
    """
static __device__ __half2 cts_hmax2(__half2 a, __half2 b) {
    union { __half2 as_h; unsigned as_bits; } x, y, out;
    x.as_h = a;
    y.as_h = b;
    unsigned lo = cts_hminmax((unsigned short)x.as_bits, (unsigned short)y.as_bits, 0);
    unsigned hi = cts_hminmax((unsigned short)(x.as_bits >> 16), (unsigned short)(y.as_bits >> 16), 0);
    out.as_bits = lo | (hi << 16);
    return out.as_h;
}
""",
)
HPRED_HELPERS["__hmin2"] = (
    "cts_hmin2",
    """
static __device__ __half2 cts_hmin2(__half2 a, __half2 b) {
    union { __half2 as_h; unsigned as_bits; } x, y, out;
    x.as_h = a;
    y.as_h = b;
    unsigned lo = cts_hminmax((unsigned short)x.as_bits, (unsigned short)y.as_bits, 1);
    unsigned hi = cts_hminmax((unsigned short)(x.as_bits >> 16), (unsigned short)(y.as_bits >> 16), 1);
    out.as_bits = lo | (hi << 16);
    return out.as_h;
}
""",
)


def hpred_arity(spelling: str) -> int:
    if spelling in {"__hneg", "__hneg2", "__habs", "__habs2", "__hisnan", "__hisnan2"}:
        return 1
    return 2


def hpred_needs_cmp(names: set[str]) -> bool:
    stems = ("cts_heq", "cts_hne", "cts_hlt", "cts_hle", "cts_hgt", "cts_hge", "cts_hb")

    def match(name: str) -> bool:
        if name.startswith("cts_hneg"):
            return False
        return name.startswith(stems)

    return any(match(name) for name in names)


def hpred_needs_minmax(names: set[str]) -> bool:
    return any(name.startswith(("cts_hmax", "cts_hmin")) for name in names)


def hpred_needs_sign(names: set[str]) -> bool:
    return any(name.startswith(("cts_hneg", "cts_habs")) for name in names)


# Measured on sm_70. lop3.b32 immediate 0xF0 copies a, 0xCC copies b, and
# 0xAA copies c, so the LUT index is (a << 2) | (b << 1) | c. bfe and bfi
# use the low 8 bits of the position and the length. A length of 0 yields 0
# from bfe and leaves the base from bfi. A start at bit 32 or above yields 0
# from bfe.u32. bfe.s32 then copies bit 31 of the source into every bit, so
# a clear sign bit stays 0. bfi leaves the base. nvcc -O2 folds a constant
# start of 32 to all ones; a start loaded from memory does not.
# bfe.s32 sign-extends from the top bit that was actually read. prmt.b32
# uses the same four nibbles as __byte_perm, and a nibble with bit 3 set
# replicates the selected byte's sign to 0x00 or 0xFF.
PTX_LOP3 = """
static __device__ unsigned cts_lop3(unsigned a, unsigned b, unsigned c, unsigned lut) {
    unsigned out = 0u;
    lut &= 0xffu;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
        if (((lut >> bit) & 1u) == 0u) continue;
        unsigned mask = 0xffffffffu;
        mask &= (bit & 4) ? a : ~a;
        mask &= (bit & 2) ? b : ~b;
        mask &= (bit & 1) ? c : ~c;
        out |= mask;
    }
    return out;
}
"""

PTX_BFE_U32 = """
static __device__ unsigned cts_bfe_u32(unsigned value, unsigned pos, unsigned len) {
    pos &= 0xffu;
    len &= 0xffu;
    if (len == 0u || pos >= 32u) return 0u;
    unsigned kept = len;
    if (pos + kept > 32u) kept = 32u - pos;
    if (kept == 32u) return value;
    return (value >> pos) & ((1u << kept) - 1u);
}
"""

PTX_BFE_S32 = """
static __device__ unsigned cts_bfe_s32(unsigned value, unsigned pos, unsigned len) {
    pos &= 0xffu;
    len &= 0xffu;
    if (len == 0u) return 0u;
    if (pos >= 32u) return (value & 0x80000000u) ? 0xffffffffu : 0u;
    unsigned kept = len;
    if (pos + kept > 32u) kept = 32u - pos;
    if (kept == 32u) return value;
    unsigned field = (value >> pos) & ((1u << kept) - 1u);
    if (((value >> (pos + kept - 1u)) & 1u) != 0u) field |= ~((1u << kept) - 1u);
    return field;
}
"""

PTX_BFI = """
static __device__ unsigned cts_bfi(unsigned src, unsigned base, unsigned pos, unsigned len) {
    pos &= 0xffu;
    len &= 0xffu;
    if (len == 0u || pos >= 32u) return base;
    unsigned kept = len;
    if (pos + kept > 32u) kept = 32u - pos;
    unsigned mask = kept == 32u ? 0xffffffffu : (((1u << kept) - 1u) << pos);
    return (base & ~mask) | ((src << pos) & mask);
}
"""

PTX_PRMT = """
static __device__ unsigned cts_prmt(unsigned a, unsigned b, unsigned selector) {
    unsigned long long source = (static_cast<unsigned long long>(b) << 32) | a;
    unsigned out = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned nibble = (selector >> (4 * i)) & 0xfu;
        unsigned byte = static_cast<unsigned>(source >> (8 * (nibble & 7u))) & 0xffu;
        if ((nibble & 8u) != 0u) byte = (byte & 0x80u) ? 0xffu : 0u;
        out |= byte << (8 * i);
    }
    return out;
}
"""


def runtime_name(spelling: str, dialect: Dialect) -> str | None:
    if dialect.prefix is None or _RUNTIME_NAME.match(spelling) is None:
        return None
    return dialect.prefix + spelling[4:]
