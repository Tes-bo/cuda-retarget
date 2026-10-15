# cuda-retarget

Translate CUDA source to HIP, to MUSA, and to the CUDA spelling MetaX compiles. This is a rewriter for people searching for a way to turn CUDA into HIP, run CUDA on AMD, or retarget a kernel toward Moore Threads (musify's source language) without switching tools per vendor.

Clang tokenises the file. The rewrite rules in this repository decide what changes. `cuda-retarget` does not invoke HIPIFY, musify, or cu-bridge, and it does not ship their source. It also does not compile. AMD's runtime, Moore Threads' runtime, MetaX's runtime, and each vendor's instruction set stay theirs. A renamed `cudaMalloc` is not a binary.

The semantic edits are the part a prefix swap drops. On HIP, a 3-argument `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, or `__shfl_down_sync` whose mask is the full 32-bit literal keeps CUDA's subgroup width of 32, because an omitted width follows the device wavefront and AMD's is 64. A use of `warpSize` becomes the literal `32` for the same reason. The name in a `#define` stays. On MUSA those calls stay 3 arguments and `warpSize` stays, because the documented warp is 32 lanes. A full-mask `__match_any_sync` or `__match_all_sync` on HIP becomes a helper that builds the mask from a subgroup of 32 lanes, comparing float and double by their IEEE bits. A full-mask `__ballot_sync`, `__any_sync`, `__all_sync`, or `__uni_sync` on HIP becomes a helper over that same subgroup, so a predicate that is uniform in one warp stays uniform there. A no-argument `__syncwarp()` on HIP becomes `__syncwarp(0xffffffff)`, which is CUDA's default mask. An explicit mask stays. On MUSA those match and vote calls stay, because the documented warp is 32 lanes. On HIP and MUSA, `__double_as_longlong`, `__longlong_as_double`, `__float_as_int`, `__int_as_float`, `__float_as_uint`, and `__uint_as_float` become device union helpers, so the IEEE bits do not depend on a CUDA intrinsic. `__double2hiint`, `__double2loint`, and `__hiloint2double` become the high and low 32-bit words of that same double. `__funnelshift_l`, `__funnelshift_lc`, `__funnelshift_r`, and `__funnelshift_rc` become 64-bit shift helpers on those same targets, so wrap versus clamp does not depend on a CUDA builtin. `__dp4a` becomes signed and unsigned four-byte dot-product helpers, so a 0x80 byte is -128 or 128 depending on the overload. `__vadd2`, `__vsub2`, `__vadd4`, `__vsub4`, and their `ss` and `us` forms become per-lane helpers: plain calls wrap, `ss` saturates as signed, and `us` saturates as unsigned. `__vmins2`, `__vminu2`, `__vmaxs2`, `__vmaxu2`, and the 4-byte forms become per-lane helpers too: `s` compares a lane as signed and `u` compares it as unsigned. `__vabs2`, `__vabs4`, `__vneg2`, `__vneg4`, and the absolute-difference forms become per-lane helpers. Plain absolute value wraps, so a minimum lane keeps its bits. The saturating absolute value turns that lane into the maximum. `__vnegss2` keeps a `0x8000` lane, the same bits `__vneg2` produces. `__vnegss4` turns `0x80` into `0x7F`. `__byte_perm` becomes a helper that concatenates the two words as `y:x` and picks four bytes. Each selector uses three bits from the low 16 bits of the third argument. `__hadd` and `__uhadd` become `(x + y) >> 1`, and `__rhadd` and `__urhadd` add one before that shift. The sum does not overflow a 32-bit temporary. `__mulhi` and `__umulhi` become the high 32 bits of a widened product, and `__mul64hi` and `__umul64hi` become the high 64 bits of a 128-bit product. A negative product sign-extends. `INT_MIN * INT_MIN` keeps the high half `2^30`. `__sad` and `__usad` become the absolute difference plus the third argument. The signed form widens first, so the absolute value of `INT_MIN` is `2^31`. The unsigned form treats those bits as a magnitude. The sum wraps modulo `2^32`. `__vsads2`, `__vsadu2`, `__vsads4`, and `__vsadu4` become the sum of the per-lane absolute differences. A signed `0xFFFF` lane is 1 and an unsigned one is 65535. A `0x8000` lane contributes 32768. The result is a scalar sum. `__vcmp*` comparisons become lane masks. A true halfword is `0xFFFF` and a true byte is `0xFF`. Signed `0x8000` is less than `0x7FFF`; the same bits unsigned are greater. MetaX keeps every CUDA spelling: that compiler accepts CUDA, and an identity translation is the honest output.

This project is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. It does not score itself inside [cuda-census](https://github.com/Tes-bo/cuda-census). cuda-census runs official toolchains only.

## Targets

| `--target` | Names | Header | 32-lane shuffles | 32-lane match and vote | Bitcast and funnel |
| --- | --- | --- | --- | --- | --- |
| `hip` | `cudaMalloc` becomes `hipMalloc`, and the same prefix rule for every `cuda` identifier whose next character is uppercase. A `warpSize` use becomes `32` | `hip/hip_runtime.h` | a full-mask 3-argument `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, or `__shfl_down_sync` gains `, 32` | full-mask match, ballot, any, all, and uni calls become helpers | union helpers, including double hi/lo words; `__funnelshift_*`, `__dp4a`, packed add/sub/min/max/abs helpers, `__byte_perm`, average helpers, high-half multiply helpers, absolute-difference helpers, packed absolute-difference sums, and packed comparison masks |
| `musa` | the same rule with the `musa` prefix | `musa_runtime.h` | unchanged | unchanged | union helpers, including double hi/lo words; `__funnelshift_*`, `__dp4a`, packed add/sub/min/max/abs helpers, `__byte_perm`, average helpers, high-half multiply helpers, absolute-difference helpers, packed absolute-difference sums, and packed comparison masks |
| `maca` | unchanged | `cuda_runtime.h` | unchanged | unchanged | unchanged |

`cudaFoo`, a user function with that prefix, is renamed on HIP and MUSA. `__shfl_sync`, `threadIdx`, and `atomicAdd` are not runtime identifiers and stay. A CUDA name inside a comment or a string literal stays. A name inside a `#define` in this file is rewritten at the definition.

## What stays source

These are not rewritten, and a later vendor compile may reject them or change their meaning:

- Inline PTX.
- A shuffle whose width argument is already a literal. That literal is kept.
- A `__syncwarp` call that already passes a mask. That mask stays.
- A 3-argument shuffle, a match call, or a `__ballot_sync` / `__any_sync` / `__all_sync` / `__uni_sync`, whose mask is not the literal `0xffffffff` (any of `u`, `l`, `ll`, `ull`). A variable mask is still CUDA's width 32 on NVIDIA and the wavefront width on AMD. This rewriter does not guess which mask the programmer meant to widen.
- `lane & 31` and any other width arithmetic.
- CUDA Graph instantiate signatures that drift between runtimes. The 5-argument call is renamed and its arguments stay.

Inserted helpers are `static __device__` and are placed after the last `#include`.

## Run

```bash
python3 -m cuda_retarget self-check
python3 -m cuda_retarget translate --target hip --input kernel.cu --output kernel.hip.cpp
python3 -m cuda_retarget translate --target musa --input kernel.cu --output kernel.mu
python3 -m cuda_retarget translate --target maca --input kernel.cu
```

Without `--output`, the translation is written to stdout. A parse failure or an unknown target exits 2. Clang must be able to load `libclang` and to see `cuda_runtime.h`. `LIBCLANG_PATH` and `CUDA_INCLUDE` override the defaults.

Host-side tests, with no vendor SDK:

```bash
python3 -m unittest discover -s tests -v
```

One test compiles the double-bits helper with `nvcc` when `nvcc` is on `PATH`. It does not compile HIP or MUSA output with `nvcc`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
