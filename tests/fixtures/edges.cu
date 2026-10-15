// Copyright 2026 Tesbo
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

// cudaSuccess stays in this comment.
__global__ void cudaFoo(int* out) {
    const char* label = "cudaMalloc";
    int lane = threadIdx.x & 31;
    int full = __shfl_sync(0xffffffff, lane, 5);
    int suffixed = __shfl_sync(0xffffffffu, lane, 5);
    int wide = __shfl_sync(0xffffffff, lane, 5, 16);
    int partial = __shfl_sync(0x0000ffff, lane, 1);
    *out = full + suffixed + wide + partial + label[0];
}
