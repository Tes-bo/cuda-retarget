# cuda-retarget

Translate CUDA source to HIP, to MUSA, and to the CUDA spelling MetaX compiles. This is a rewriter for people searching for a way to turn CUDA into HIP, run CUDA on AMD, or retarget a kernel toward Moore Threads (musify's source language) without switching tools per vendor.

Clang tokenises the file. The rewrite rules in this repository decide what changes. `cuda-retarget` does not invoke HIPIFY, musify, or cu-bridge, and it does not ship their source. It also does not compile. AMD's runtime, Moore Threads' runtime, MetaX's runtime, and each vendor's instruction set stay theirs. A renamed `cudaMalloc` is not a binary.

The semantic edits are the part a prefix swap drops. On HIP, a 3-argument `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, or `__shfl_down_sync` whose mask is the full 32-bit literal keeps CUDA's subgroup width of 32, because an omitted width follows the device wavefront and AMD's is 64. A use of `warpSize` becomes the literal `32` for the same reason. The name in a `#define` stays. On MUSA those calls stay 3 arguments and `warpSize` stays, because the documented warp is 32 lanes. A full-mask `__match_any_sync` or `__match_all_sync` on HIP becomes a helper that builds the mask from a subgroup of 32 lanes, comparing float and double by their IEEE bits. On MUSA those match calls stay, because the documented warp is 32 lanes. On HIP and MUSA, `__double_as_longlong`, `__longlong_as_double`, `__float_as_int`, `__int_as_float`, `__float_as_uint`, and `__uint_as_float` become device union helpers, so the IEEE bits do not depend on a CUDA intrinsic. MetaX keeps every CUDA spelling: that compiler accepts CUDA, and an identity translation is the honest output.

This project is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. It does not score itself inside [cuda-census](https://github.com/Tes-bo/cuda-census). cuda-census runs official toolchains only.

## Targets

| `--target` | Names | Header | 32-lane shuffles | 32-lane match | Bitcast intrinsics |
| --- | --- | --- | --- | --- | --- |
| `hip` | `cudaMalloc` becomes `hipMalloc`, and the same prefix rule for every `cuda` identifier whose next character is uppercase. A `warpSize` use becomes `32` | `hip/hip_runtime.h` | a full-mask 3-argument `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, or `__shfl_down_sync` gains `, 32` | full-mask `__match_any_sync` and `__match_all_sync` become helpers | union helpers |
| `musa` | the same rule with the `musa` prefix | `musa_runtime.h` | unchanged | unchanged | union helpers |
| `maca` | unchanged | `cuda_runtime.h` | unchanged | unchanged | unchanged |

`cudaFoo`, a user function with that prefix, is renamed on HIP and MUSA. `__shfl_sync`, `threadIdx`, and `atomicAdd` are not runtime identifiers and stay. A CUDA name inside a comment or a string literal stays. A name inside a `#define` in this file is rewritten at the definition.

## What stays source

These are not rewritten, and a later vendor compile may reject them or change their meaning:

- Inline PTX.
- A shuffle whose width argument is already a literal. That literal is kept.
- A 3-argument shuffle, or a `__match_any_sync` / `__match_all_sync`, whose mask is not the literal `0xffffffff` (any of `u`, `l`, `ll`, `ull`). A variable mask is still CUDA's width 32 on NVIDIA and the wavefront width on AMD. This rewriter does not guess which mask the programmer meant to widen.
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
