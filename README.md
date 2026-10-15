# cuda-retarget

Turn one CUDA file into HIP, MUSA, or the CUDA spelling MetaX compiles. The rewriter in this repository decides the edits. It does not call HIPIFY, musify, or cu-bridge, and it does not ship their source.

[![License](https://img.shields.io/github/license/Tes-bo/cuda-retarget?style=flat-square&color=blue)](LICENSE)
[![Dialects](https://img.shields.io/badge/dialects-HIP%20%C2%B7%20MUSA%20%C2%B7%20MACA-1a7f37?style=flat-square)](#targets)
[![Python](https://img.shields.io/badge/python-3.10%2B-0969da?style=flat-square)](#quick-start)

[cuda-census](https://github.com/Tes-bo/cuda-census) runs the official toolchains and records their numbers. This repository does not score itself there.

<p align="center">
  <img src="docs/overview.svg" alt="CUDA source is parsed once by Clang and emitted as HIP, MUSA, or unchanged MetaX CUDA." width="820">
</p>

**Quick start · [What changes](#what-changes) · [Targets](#targets) · [Edges](#edges-the-helpers-match) · [Run](#run)**

A renamed `cudaMalloc` is source, not a binary. AMD's runtime, Moore Threads' runtime, MetaX's runtime, and each vendor's instruction set stay theirs. This project is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX.

## Quick start

Python 3.10 or newer. Clang must load `libclang` and see `cuda_runtime.h`. `LIBCLANG_PATH` and `CUDA_INCLUDE` override the defaults.

```bash
python3 -m cuda_retarget self-check
python3 -m cuda_retarget translate --target hip --input kernel.cu --output kernel.hip.cpp
python3 -m cuda_retarget translate --target musa --input kernel.cu --output kernel.mu
python3 -m cuda_retarget translate --target maca --input kernel.cu
```

Without `--output`, the translation goes to stdout. A parse failure or an unknown target exits 2.

## What changes

The sample the self-check translates:

```cuda
*out = __shfl_sync(0xffffffff, threadIdx.x, 0) + warpSize;
cudaMalloc(&p, sizeof(int));
```

HIP keeps CUDA's subgroup of 32 lanes. An omitted width would follow the device wavefront, and AMD's is 64.

```cpp
*out = __shfl_sync(0xffffffff, threadIdx.x, 0, 32) + 32;
hipMalloc(&p, sizeof(int));
```

MUSA keeps the 3-argument shuffle and `warpSize`. The documented warp there is 32 lanes. The runtime prefix still changes.

```cpp
*out = __shfl_sync(0xffffffff, threadIdx.x, 0) + warpSize;
musaMalloc(&p, sizeof(int));
```

MACA is the same text as the input. That compiler accepts CUDA, so an identity translation is the honest output.

Comments and string literals stay. A name inside a `#define` is rewritten at the definition. The `#define` name `warpSize` itself stays; a later use of it does not. `cudaFoo`, a user function with that prefix, is renamed on HIP and MUSA. `__shfl_sync`, `threadIdx`, and `atomicAdd` are not runtime identifiers.

## Targets

| `--target` | Names | Header | 32-lane warp | Integer and float helpers |
| --- | --- | --- | --- | --- |
| `hip` | `cudaMalloc` becomes `hipMalloc`, and the same prefix rule for every `cuda` identifier whose next character is uppercase. A `warpSize` use becomes `32` | `hip/hip_runtime.h` | a full-mask 3-argument shuffle gains `, 32`. Full-mask match, ballot, any, all, and uni become 32-lane helpers. `__syncwarp()` becomes `__syncwarp(0xffffffff)` | lowered |
| `musa` | the same rule with the `musa` prefix | `musa_runtime.h` | unchanged | lowered |
| `maca` | unchanged | `cuda_runtime.h` | unchanged | unchanged |

The helpers are bitcasts and double hi/lo words, funnel shifts, `__dp4a`, packed add/sub/min/max/abs, `__byte_perm`, averages, high-half products, absolute differences, packed absolute-difference sums, comparison masks, lane predicates, packed averages, bit scans, 24-bit products, rounding conversions, double-to-float rounding, `__saturatef`, and directed float add, sub, mul, and fused multiply-add.

Inserted helpers are `static __device__` and sit after the last `#include`.

## Edges the helpers match

These are the results a prefix swap gets wrong. HIP and MUSA both get the helpers. Warp-width edits are HIP only.

| Call | Result |
| --- | --- |
| full-mask `__shfl_sync` / `xor` / `up` / `down` with 3 arguments, on HIP | `, 32` inserted. A width that is already present stays, including `warpSize` |
| `warpSize` use, on HIP | literal `32`. The name in `#define warpSize` stays |
| full-mask `__match_any_sync` / `__match_all_sync`, on HIP | a 32-lane mask. Float and double are compared by IEEE bits, so `-0` and `+0` differ |
| full-mask `__ballot_sync` / `__any_sync` / `__all_sync` / `__uni_sync`, on HIP | the same 32-lane subgroup. A predicate that is uniform in one warp stays uniform there |
| `__syncwarp()`, on HIP | `__syncwarp(0xffffffff)`. An explicit mask stays |
| `__vnegss2` of a `0x8000` lane | stays `0x8000`, the same bits as `__vneg2`. `__vnegss4` turns `0x80` into `0x7F` |
| `__vset*` | a true halfword writes `0x0001` and a true byte writes `0x01`. Failing lanes stay zero |
| `__vavgs2(1, 2)` and `__vavgs2(-2, -3)` | `2` and `-3`. Half rounds away from zero |
| `__vhaddu2(0xFFFF, 1)` | `0x8000`, not a wrapped `0` |
| `__hadd(INT_MIN, INT_MIN)` | `INT_MIN`. The average of `0xFFFFFFFF` and `1` is `2^31` |
| `INT_MIN * INT_MIN` through `__mulhi` | high half `2^30`, not a wrapped 32-bit product |
| `__sad` of `INT_MIN` | absolute value `2^31`. The following add wraps modulo `2^32` |
| `__vsad*` | the scalar sum of the lane differences. A `0x8000` lane contributes `32768` |
| `__vcmp*` | a true halfword is `0xFFFF` and a true byte is `0xFF` |
| `__float2int_rn(2.5f)` / `__float2int_ru(2.5f)` / `__float2int_rd(-2.5f)` | `2` / `3` / `-3`. Out-of-range values saturate |
| NaN to `__float2int_*` | `0` |
| NaN to `__float2ll_*` or `__float2ull_*` | the high bit |
| NaN to `__double2int_*` | `0x80000000`, in every rounding mode |
| `__double2float_rn(1+2^-24)` / `__double2float_ru` of that tie | stays on 1 / the next float. Toward zero, a value past the finite maximum stays finite |
| a double NaN passed to `__double2float_*` | a quiet float NaN with the same sign. `-0` stays `-0` |
| `__saturatef` of a NaN or of `-0` | `+0`. Values outside `[0, 1]` clamp |
| `__fadd_ru` of 1 and a value below one ulp | the next float above 1. The nearest-even tie stays on the even float |
| `__fadd_rd(+0, -0)` / `__fadd_rd(+0, +0)` | `-0` / `+0`. Same-sign zeros keep their sign |
| `__fadd_rz` of two overflowing magnitudes | the finite maximum, not infinity |
| `__fmaf_rn` of `(1+2^-23)` squared minus the rounded square | `2^-46` (`679477248`), not 0 |
| a NaN operand of `__fadd_*` or `__fmaf_*` | the canonical quiet NaN `2147483647` |
| `__mul24` | the high 8 bits are ignored. The square of the minimum 24-bit value keeps a low half of `0` |
| `__clz(0)` / `__clzll(0)` / `__ffs(0)` | `32` / `64` / `0` |
| `__dp4a` of a `0x80` byte | `-128` signed, `128` unsigned |
| `__byte_perm` | four bytes from `y:x`. Selector bits above bit 15 are ignored |
| `__funnelshift_l` / `_r` | `shift & 31`. The clamp forms stop at 32 |

Signed lane `0x8000` and unsigned lane `0x8000` are different numbers in the packed comparisons, min/max, and absolute differences.

## What stays source

A later vendor compile may reject these, or give them a different meaning:

- Inline PTX.
- A shuffle that already has a width argument. That literal is kept.
- A `__syncwarp` call that already passes a mask.
- A 3-argument shuffle, a match call, or a vote call whose mask is not the literal `0xffffffff` (`u`, `l`, `ll`, or `ull`). A variable mask is still CUDA's width 32 on NVIDIA and the wavefront width on AMD. This rewriter does not guess which mask the programmer meant.
- `lane & 31` and any other width arithmetic.
- The 5-argument CUDA Graph instantiate call. It is renamed, and its arguments stay. The `0` there is a buffer size, not flags.

## Run

```bash
python3 -m unittest discover -s tests -v
```

The tests do not compile HIP or MUSA output with `nvcc`. When `nvcc` is on `PATH` and a GPU is free, they compile the helper text and compare it with the CUDA intrinsic.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
