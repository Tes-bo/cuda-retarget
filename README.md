# cuda-retarget

Translate CUDA source to HIP, to MUSA, and to the CUDA spelling MetaX compiles. This is a rewriter for people searching for a way to turn CUDA into HIP, run CUDA on AMD, or retarget a kernel toward Moore Threads (musify's source language) without switching tools per vendor.

Clang tokenises the file. The rewrite rules in this repository decide what changes. `cuda-retarget` does not invoke HIPIFY, musify, or cu-bridge, and it does not ship their source. It also does not compile. AMD's runtime, Moore Threads' runtime, MetaX's runtime, and each vendor's instruction set stay theirs. A renamed `cudaMalloc` is not a binary.

The semantic edits are the part a prefix swap drops. On HIP, a 3-argument `__shfl_sync` whose mask is the full 32-bit literal keeps CUDA's subgroup width of 32, because an omitted width follows the device wavefront and AMD's is 64. On MUSA that call stays 3 arguments, because the documented warp is 32 lanes. `__double_as_longlong` becomes a device union helper on HIP and MUSA so the IEEE bits of the value do not depend on a CUDA intrinsic. MetaX keeps every CUDA spelling: that compiler accepts CUDA, and an identity translation is the honest output.

This project is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. It does not score itself inside [cuda-census](https://github.com/Tes-bo/cuda-census). cuda-census runs official toolchains only.

## Targets

| `--target` | Names | Header | 32-lane `__shfl_sync` | `__double_as_longlong` |
| --- | --- | --- | --- | --- |
| `hip` | `cudaMalloc` becomes `hipMalloc`, and the same prefix rule for every `cuda` identifier whose next character is uppercase | `hip/hip_runtime.h` | a full-mask 3-argument call gains `, 32` | `cts_double_bits` |
| `musa` | the same rule with the `musa` prefix | `musa_runtime.h` | unchanged | `cts_double_bits` |
| `maca` | unchanged | `cuda_runtime.h` | unchanged | unchanged |

`cudaFoo`, a user function with that prefix, is renamed on HIP and MUSA. `__shfl_sync`, `threadIdx`, and `atomicAdd` are not runtime identifiers and stay. A CUDA name inside a comment or a string literal stays. A name inside a `#define` in this file is rewritten at the definition.

## What stays source

These are not rewritten, and a later vendor compile may reject them or change their meaning:

- Inline PTX.
- A `__shfl_sync` whose width is already written. That width is kept, including `warpSize`.
- A 3-argument `__shfl_sync` whose mask is not the literal `0xffffffff` (any of `u`, `l`, `ll`, `ull`). A variable mask is still CUDA's width 32 on NVIDIA and the wavefront width on AMD. This rewriter does not guess which mask the programmer meant to widen.
- `lane & 31` and any other width arithmetic.
- CUDA Graph instantiate signatures that drift between runtimes. The 5-argument call is renamed and its arguments stay.

The helper inserted for the double-bits intrinsic is `static __device__` and is placed after the last `#include`.

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
