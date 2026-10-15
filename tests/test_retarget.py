# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Rewriter checks. They do not call vendor compilers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_retarget.dialect import (
    BITCAST_HELPERS,
    DOUBLE_WORD_HELPERS,
    DP4A_HELPER,
    FUNNEL_HELPERS,
    MATCH_ALL_HELPER,
    MATCH_ANY_HELPER,
    VOTE_HELPERS,
)
from cuda_retarget.translate import TranslateError, translate_path, translate_source

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "edges.cu"
CENSUS = Path(__file__).resolve().parents[2] / "cuda-census" / "corpus"
BITS = 4607182418800017408
FLOAT_BITS = 1065353216
BITCAST_SOURCE = """\
#include <cuda_runtime.h>
__global__ void convert(int* i, unsigned* u, float* f, double* d, long long* ll) {
    const char* label = "__float_as_int";
    *i = __float_as_int(*f);
    *u = __float_as_uint(*f);
    *f = __int_as_float(*i);
    *f = __uint_as_float(*u);
    *ll = __double_as_longlong(*d);
    *d = __longlong_as_double(*ll);
    *i += label[0];
}
"""


class EdgeTests(unittest.TestCase):
    def test_hip_keeps_literals_and_explicit_widths(self) -> None:
        hip = translate_path(FIXTURE, "hip")
        self.assertIn("// cudaSuccess stays in this comment.", hip)
        self.assertIn('"cudaMalloc"', hip)
        self.assertIn("threadIdx.x & 31", hip)
        self.assertIn("__shfl_sync(0xffffffff, lane, 5, 32)", hip)
        self.assertIn("__shfl_sync(0xffffffffu, lane, 5, 32)", hip)
        self.assertIn("__shfl_sync(0xffffffff, lane, 5, 16)", hip)
        self.assertIn("__shfl_sync(0x0000ffff, lane, 1)", hip)
        self.assertNotIn("__shfl_sync(0xffffffff, lane, 5, 16, 32)", hip)
        self.assertIn("__shfl_xor_sync(0xffffffff, lane, 1, 32)", hip)
        self.assertIn("__shfl_up_sync(0xffffffff, lane, 1, 32)", hip)
        self.assertIn("__shfl_down_sync(0xffffffff, lane, 1, 32)", hip)
        self.assertIn("#define warpSize 32", hip)
        self.assertIn("int width = 32;", hip)
        self.assertIn("hipFoo", hip)
        self.assertNotIn("cudaFoo", hip)
        self.assertIn("<hip/hip_runtime.h>", hip)

    def test_musa_does_not_pin_shuffle_width(self) -> None:
        musa = translate_path(FIXTURE, "musa")
        self.assertIn("__shfl_sync(0xffffffff, lane, 5);", musa)
        self.assertIn("__shfl_xor_sync(0xffffffff, lane, 1);", musa)
        self.assertIn("int width = warpSize;", musa)
        self.assertNotIn(", 32)", musa)
        self.assertIn("musaFoo", musa)
        self.assertIn("<musa_runtime.h>", musa)

    def test_maca_is_the_cuda_spelling(self) -> None:
        source = FIXTURE.read_text(encoding="utf-8")
        self.assertEqual(translate_path(FIXTURE, "maca"), source)

    def test_unknown_target_is_refused(self) -> None:
        with self.assertRaises(TranslateError):
            translate_source("int main(){return 0;}\n", "spirv")

    def test_only_hip_lowers_full_mask_match(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void match_keys(unsigned* out) {
    const char* label = "__match_any_sync";
    unsigned key = threadIdx.x & 31u;
    int pred = 0;
    unsigned partial = __match_any_sync(0x0000ffff, key);
    unsigned full = __match_any_sync(0xffffffff, key);
    unsigned all = __match_all_sync(0xffffffff, key, &pred);
    *out = partial + full + all + pred + label[0];
}
"""
        hip = translate_source(source, "hip")
        self.assertIn("__match_any_sync(0x0000ffff, key)", hip)
        self.assertIn("cts_match_any_sync(0xffffffff, key)", hip)
        self.assertIn("cts_match_all_sync(0xffffffff, key, &pred)", hip)
        self.assertIn('"__match_any_sync"', hip)
        self.assertIn("__shfl_sync(mask, value, src, 32)", hip)
        musa = translate_source(source, "musa")
        self.assertIn("__match_any_sync(0xffffffff, key)", musa)
        self.assertNotIn("cts_match_any_sync", musa)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_only_hip_writes_the_syncwarp_mask(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void exchange(int* out) {
    const char* label = "__syncwarp";
    __syncwarp();
    __syncwarp(0x0000ffff);
    *out = label[0];
}
"""
        hip = translate_source(source, "hip")
        self.assertIn("__syncwarp(0xffffffff)", hip)
        self.assertIn("__syncwarp(0x0000ffff)", hip)
        self.assertNotIn("__syncwarp()", hip)
        self.assertIn('"__syncwarp"', hip)
        musa = translate_source(source, "musa")
        self.assertIn("__syncwarp()", musa)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_only_hip_lowers_full_mask_vote(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void vote(unsigned* out) {
    const char* label = "__ballot_sync";
    int pred = threadIdx.x < 32;
    unsigned partial = __ballot_sync(0x0000ffff, pred);
    unsigned full = __ballot_sync(0xffffffff, pred);
    int any = __any_sync(0xffffffff, pred);
    int all = __all_sync(0xffffffff, pred);
    int uni = __uni_sync(0xffffffff, pred);
    *out = partial + full + any + all + uni + label[0];
}
"""
        hip = translate_source(source, "hip")
        self.assertIn("__ballot_sync(0x0000ffff, pred)", hip)
        self.assertIn("cts_ballot_sync(0xffffffff, pred)", hip)
        self.assertIn("cts_any_sync(0xffffffff, pred)", hip)
        self.assertIn("cts_all_sync(0xffffffff, pred)", hip)
        self.assertIn("cts_uni_sync(0xffffffff, pred)", hip)
        self.assertIn('"__ballot_sync"', hip)
        self.assertIn("__shfl_sync(mask, predicate, src, 32)", hip)
        musa = translate_source(source, "musa")
        self.assertIn("__ballot_sync(0xffffffff, pred)", musa)
        self.assertNotIn("cts_ballot_sync", musa)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_funnelshifts(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void shift(unsigned* out) {
    const char* label = "__funnelshift_l";
    unsigned lo = 0xA5A5A5A5u;
    unsigned hi = 0x3C3C3C3Cu;
    out[0] = __funnelshift_l(lo, hi, 1);
    out[1] = __funnelshift_lc(lo, hi, 32);
    out[2] = __funnelshift_r(lo, hi, 1);
    out[3] = __funnelshift_rc(lo, hi, 32);
    out[4] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__funnelshift_l"', translated)
            for intrinsic, (name, _helper) in FUNNEL_HELPERS.items():
                self.assertNotIn(intrinsic + "(", translated)
                self.assertIn(name + "(", translated)
                self.assertEqual(translated.count(name + "("), 2)
            self.assertIn("shift & 31u", translated)
            self.assertIn("shift > 32u", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_bitcasts(self) -> None:
        for target in ("hip", "musa"):
            translated = translate_source(BITCAST_SOURCE, target)
            self.assertIn('"__float_as_int"', translated)
            for intrinsic, (name, _helper) in BITCAST_HELPERS.items():
                self.assertNotIn(intrinsic + "(", translated)
                self.assertIn(name + "(", translated)
                self.assertEqual(translated.count(name + "("), 2)
        self.assertEqual(translate_source(BITCAST_SOURCE, "maca"), BITCAST_SOURCE)

    def test_hip_and_musa_lower_double_words(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void split(int* hi, int* lo, double* value) {
    const char* label = "__double2hiint";
    *hi = __double2hiint(*value);
    *lo = __double2loint(*value);
    *value = __hiloint2double(*hi, *lo);
    *hi += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__double2hiint"', translated)
            self.assertIn("cts_double_hiint(*value)", translated)
            self.assertIn("cts_double_loint(*value)", translated)
            self.assertIn("cts_hiloint_double(*hi, *lo)", translated)
            self.assertNotIn("__double2hiint(", translated)
            self.assertNotIn("__double2loint(", translated)
            self.assertNotIn("__hiloint2double(", translated)
            for _intrinsic, (name, _helper) in DOUBLE_WORD_HELPERS.items():
                self.assertEqual(translated.count(name + "("), 2)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_dp4a(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void dot4(int* s, unsigned* u) {
    const char* label = "__dp4a";
    int a = 0x0101FF01;
    int b = 0x01010101;
    unsigned ua = 0x0101FF01u;
    unsigned ub = 0x01010101u;
    *s = __dp4a(a, b, 0);
    *u = __dp4a(ua, ub, 0u);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__dp4a"', translated)
            self.assertIn("cts_dp4a(a, b, 0)", translated)
            self.assertIn("cts_dp4a(ua, ub, 0u)", translated)
            self.assertNotIn("__dp4a(", translated)
            self.assertIn("signed char", translated)
            self.assertIn("0xffu", translated)
            self.assertEqual(translated.count("cts_dp4a("), 4)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_unparsed_source_is_refused(self) -> None:
        with self.assertRaises(TranslateError):
            translate_source("int main( {\n", "hip", filename="broken.cu")


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        if not CENSUS.is_dir():
            self.skipTest("cuda-census corpus is not beside this repo")

    def test_maca_matches_every_case(self) -> None:
        for path in sorted(CENSUS.glob("*/main.cu")):
            source = path.read_text(encoding="utf-8")
            self.assertEqual(translate_path(path, "maca"), source, path.name)

    def test_hip_and_musa_rename_the_runtime(self) -> None:
        path = CENSUS / "vector_add_int" / "main.cu"
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        self.assertIn("hipMalloc", hip)
        self.assertIn("<<<", hip)
        self.assertNotIn("cudaMalloc", hip)
        self.assertIn("threadIdx", hip)
        self.assertIn("musaMalloc", musa)
        self.assertNotIn("cudaMalloc", musa)

    def test_only_hip_pins_the_census_shuffle(self) -> None:
        path = CENSUS / "warp_shuffle" / "main.cu"
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        self.assertIn("__shfl_sync(0xffffffff, lane, 5, 32)", hip)
        self.assertIn("__shfl_sync(0xffffffff, lane, 5);", musa)
        self.assertNotIn(", 32)", musa)

    def test_hip_pins_the_census_xor_reduce(self) -> None:
        path = CENSUS / "warp_reduce" / "main.cu"
        if not path.is_file():
            self.skipTest("warp_reduce is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        self.assertEqual(hip.count("__shfl_xor_sync(0xffffffff, value, "), 5)
        self.assertIn("__shfl_xor_sync(0xffffffff, value, 16, 32)", hip)
        self.assertIn("__shfl_xor_sync(0xffffffff, value, 1, 32)", hip)
        self.assertIn("__shfl_xor_sync(0xffffffff, value, 16);", musa)
        self.assertNotIn(", 32)", musa)

    def test_hip_and_musa_lower_the_census_funnelshift(self) -> None:
        path = CENSUS / "funnelshift" / "main.cu"
        if not path.is_file():
            self.skipTest("funnelshift is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_funnelshift_l(lo, hi, 1)", translated)
            self.assertIn("cts_funnelshift_l(lo, hi, 32)", translated)
            self.assertIn("cts_funnelshift_lc(lo, hi, 32)", translated)
            self.assertIn("cts_funnelshift_r(lo, hi, 1)", translated)
            self.assertIn("cts_funnelshift_rc(lo, hi, 32)", translated)
            self.assertNotIn("__funnelshift_l(", translated)
        self.assertIn("__funnelshift_l", maca)
        self.assertNotIn("cts_funnelshift_l", maca)

    def test_hip_and_musa_lower_the_census_dp4a(self) -> None:
        path = CENSUS / "dp4a" / "main.cu"
        if not path.is_file():
            self.skipTest("dp4a is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_dp4a(0x01010101, 0x01010101, 0)", translated)
            self.assertIn("cts_dp4a(a, b, 0)", translated)
            self.assertIn("cts_dp4a(ua, ub, 0u)", translated)
            self.assertNotIn("__dp4a(", translated)
        self.assertIn("__dp4a", maca)
        self.assertNotIn("cts_dp4a", maca)

    def test_only_hip_pins_warp_size(self) -> None:
        path = CENSUS / "warp_size" / "main.cu"
        if not path.is_file():
            self.skipTest("warp_size is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        self.assertIn("*out = 32;", hip)
        self.assertNotIn("warpSize", hip)
        self.assertIn("*out = warpSize;", musa)

    def test_double_bits_helper_is_hip_and_musa_only(self) -> None:
        path = CENSUS / "double_as_longlong" / "main.cu"
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_double_bits(1.0)", translated)
            self.assertIn("union", translated)
            self.assertNotIn("__double_as_longlong", translated)
            self.assertLess(translated.index("cts_double_bits"), translated.index("bits_of_one"))
        self.assertIn("__double_as_longlong", maca)
        self.assertNotIn("cts_double_bits", maca)

    def test_hip_and_musa_lower_the_census_double_words(self) -> None:
        path = CENSUS / "double_hi_lo" / "main.cu"
        if not path.is_file():
            self.skipTest("double_hi_lo is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_double_hiint(1.0)", translated)
            self.assertIn("cts_double_loint(1.0)", translated)
            self.assertIn("cts_hiloint_double(hi, lo)", translated)
            self.assertNotIn("__double2hiint(", translated)
            self.assertNotIn("__hiloint2double(", translated)
        self.assertIn("__double2hiint", maca)
        self.assertNotIn("cts_double_hiint", maca)

    def test_graph_instantiate_keeps_five_arguments(self) -> None:
        path = CENSUS / "graph_async_alloc" / "main.cu"
        hip = translate_path(path, "hip")
        self.assertIn("hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0)", hip)
        self.assertIn("hipMallocAsync", hip)
        self.assertIn("hipFreeAsync", hip)


def _idle_gpu() -> str | None:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return "0"
    listed = subprocess.run(
        [smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return None
    busy = subprocess.run(
        [smi, "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if busy.returncode != 0:
        return None
    # A compute process makes every index with used memory above the driver
    # baseline a poor place to run. Prefer an index whose memory is still idle.
    if busy.stdout.strip():
        for line in listed.stdout.splitlines():
            index, used = [part.strip() for part in line.split(",")]
            if int(used) < 64:
                return index
        return None
    first = listed.stdout.splitlines()
    if not first:
        return None
    return first[0].split(",")[0].strip()


class HelperBitsTests(unittest.TestCase):
    def test_union_helper_matches_the_ieee_bits_of_one(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(
            BITCAST_HELPERS[name][1].strip("\n")
            for name in ("__double_as_longlong", "__float_as_int")
        )
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void bits_of_one(unsigned long long* out_d, int* out_f) {{
    *out_d = cts_double_bits(1.0);
    *out_f = cts_float_bits(1.0f);
}}
int main() {{
    unsigned long long* out_d = nullptr;
    int* out_f = nullptr;
    if (cudaMalloc(&out_d, sizeof(*out_d)) != cudaSuccess) return 2;
    if (cudaMalloc(&out_f, sizeof(*out_f)) != cudaSuccess) return 2;
    bits_of_one<<<1, 1>>>(out_d, out_f);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    unsigned long long bits = 0;
    int fbits = 0;
    if (cudaMemcpy(&bits, out_d, sizeof(bits), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    if (cudaMemcpy(&fbits, out_f, sizeof(fbits), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("bits=%llu\\n", bits);
    std::printf("fbits=%d\\n", fbits);
    cudaFree(out_d);
    cudaFree(out_f);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bits.cu"
            binary = Path(directory) / "bits"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), f"bits={BITS}\nfbits={FLOAT_BITS}")

    def test_match_helper_agrees_with_the_intrinsic(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = MATCH_ANY_HELPER.strip("\n") + "\n" + MATCH_ALL_HELPER.strip("\n")
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare_keys(int* bad, unsigned* lane0) {{
    unsigned key = threadIdx.x & 31u;
    int pred_i = 0;
    int pred_h = 0;
    unsigned any_i = __match_any_sync(0xffffffff, key);
    unsigned any_h = cts_match_any_sync(0xffffffff, key);
    unsigned all_i = __match_all_sync(0xffffffff, key, &pred_i);
    unsigned all_h = cts_match_all_sync(0xffffffff, key, &pred_h);
    if (any_i != any_h || all_i != all_h || pred_i != pred_h)
        atomicExch(bad, 1);
    if (threadIdx.x == 0)
        *lane0 = any_i;
}}
__global__ void compare_zeros(int* bad) {{
    float value = threadIdx.x == 0 ? -0.f : 0.f;
    unsigned any_i = __match_any_sync(0xffffffff, value);
    unsigned any_h = cts_match_any_sync(0xffffffff, value);
    if (any_i != any_h)
        atomicExch(bad, 1);
}}
int main() {{
    int* bad = nullptr;
    unsigned* lane0 = nullptr;
    if (cudaMalloc(&bad, sizeof(*bad)) != cudaSuccess) return 2;
    if (cudaMalloc(&lane0, sizeof(*lane0)) != cudaSuccess) return 2;
    if (cudaMemset(bad, 0, sizeof(*bad)) != cudaSuccess) return 2;
    compare_keys<<<1, 64>>>(bad, lane0);
    compare_zeros<<<1, 32>>>(bad);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    int mismatch = 0;
    unsigned lane = 0;
    if (cudaMemcpy(&mismatch, bad, sizeof(mismatch), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    if (cudaMemcpy(&lane, lane0, sizeof(lane), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("mismatch=%d\\n", mismatch);
    std::printf("lane0=%u\\n", lane);
    cudaFree(bad);
    cudaFree(lane0);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.cu"
            binary = Path(directory) / "match"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), "mismatch=0\nlane0=1")

    def test_vote_helper_agrees_with_the_intrinsic(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(text.strip("\n") for text in VOTE_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__device__ int vote_mismatch(int predicate) {{
    unsigned ballot_i = __ballot_sync(0xffffffff, predicate);
    unsigned ballot_h = cts_ballot_sync(0xffffffff, predicate);
    int any_i = __any_sync(0xffffffff, predicate);
    int any_h = cts_any_sync(0xffffffff, predicate);
    int all_i = __all_sync(0xffffffff, predicate);
    int all_h = cts_all_sync(0xffffffff, predicate);
    int uni_i = __uni_sync(0xffffffff, predicate);
    int uni_h = cts_uni_sync(0xffffffff, predicate);
    return ballot_i != ballot_h || any_i != any_h || all_i != all_h || uni_i != uni_h;
}}
__global__ void compare_split(int* bad) {{
    if (vote_mismatch(threadIdx.x < 32))
        atomicExch(bad, 1);
}}
__global__ void compare_alt(int* bad) {{
    if (vote_mismatch(threadIdx.x & 1))
        atomicExch(bad, 1);
}}
int main() {{
    int* bad = nullptr;
    if (cudaMalloc(&bad, sizeof(*bad)) != cudaSuccess) return 2;
    if (cudaMemset(bad, 0, sizeof(*bad)) != cudaSuccess) return 2;
    compare_split<<<1, 64>>>(bad);
    compare_alt<<<1, 32>>>(bad);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    int mismatch = 0;
    if (cudaMemcpy(&mismatch, bad, sizeof(mismatch), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("mismatch=%d\\n", mismatch);
    cudaFree(bad);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vote.cu"
            binary = Path(directory) / "vote"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), "mismatch=0")

    def test_funnel_helper_agrees_with_the_intrinsic(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in FUNNEL_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned lo = 0xA5A5A5A5u;
    unsigned hi = 0x3C3C3C3Cu;
    if (__funnelshift_l(lo, hi, 1) != cts_funnelshift_l(lo, hi, 1))
        atomicExch(bad, 1);
    if (__funnelshift_l(lo, hi, 32) != cts_funnelshift_l(lo, hi, 32))
        atomicExch(bad, 1);
    if (__funnelshift_lc(lo, hi, 32) != cts_funnelshift_lc(lo, hi, 32))
        atomicExch(bad, 1);
    if (__funnelshift_lc(lo, hi, 33) != cts_funnelshift_lc(lo, hi, 33))
        atomicExch(bad, 1);
    if (__funnelshift_r(lo, hi, 1) != cts_funnelshift_r(lo, hi, 1))
        atomicExch(bad, 1);
    if (__funnelshift_rc(lo, hi, 32) != cts_funnelshift_rc(lo, hi, 32))
        atomicExch(bad, 1);
    if (__funnelshift_rc(lo, hi, 33) != cts_funnelshift_rc(lo, hi, 33))
        atomicExch(bad, 1);
}}
int main() {{
    int* bad = nullptr;
    if (cudaMalloc(&bad, sizeof(*bad)) != cudaSuccess) return 2;
    if (cudaMemset(bad, 0, sizeof(*bad)) != cudaSuccess) return 2;
    compare<<<1, 1>>>(bad);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    int mismatch = 0;
    if (cudaMemcpy(&mismatch, bad, sizeof(mismatch), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("mismatch=%d\\n", mismatch);
    cudaFree(bad);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "funnel.cu"
            binary = Path(directory) / "funnel"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), "mismatch=0")

    def test_double_word_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in DOUBLE_WORD_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__device__ int words_mismatch(double value) {{
    int hi_i = __double2hiint(value);
    int lo_i = __double2loint(value);
    int hi_h = cts_double_hiint(value);
    int lo_h = cts_double_loint(value);
    if (hi_i != hi_h || lo_i != lo_h)
        return 1;
    double again_i = __hiloint2double(hi_i, lo_i);
    double again_h = cts_hiloint_double(hi_h, lo_h);
    return __double2hiint(again_i) != cts_double_hiint(again_h)
        || __double2loint(again_i) != cts_double_loint(again_h);
}}
__global__ void compare(int* bad) {{
    double values[5] = {{0.0, -0.0, 1.0, -1.0, 3.141592653589793}};
    for (int i = 0; i < 5; ++i) {{
        if (words_mismatch(values[i]))
            atomicExch(bad, 1);
    }}
}}
int main() {{
    int* bad = nullptr;
    if (cudaMalloc(&bad, sizeof(*bad)) != cudaSuccess) return 2;
    if (cudaMemset(bad, 0, sizeof(*bad)) != cudaSuccess) return 2;
    compare<<<1, 1>>>(bad);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    int mismatch = 0;
    if (cudaMemcpy(&mismatch, bad, sizeof(mismatch), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("mismatch=%d\\n", mismatch);
    cudaFree(bad);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.cu"
            binary = Path(directory) / "words"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), "mismatch=0")

    def test_dp4a_helper_agrees_with_the_intrinsic(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = DP4A_HELPER.strip("\n")
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    int a = 0x0101FF01;
    int b = 0x01010101;
    unsigned ua = 0x0101FF01u;
    unsigned ub = 0x01010101u;
    if (__dp4a(0x01010101, 0x01010101, 0) != cts_dp4a(0x01010101, 0x01010101, 0))
        atomicExch(bad, 1);
    if (__dp4a(a, b, 0) != cts_dp4a(a, b, 0))
        atomicExch(bad, 1);
    if (__dp4a(a, b, 4) != cts_dp4a(a, b, 4))
        atomicExch(bad, 1);
    if (__dp4a((int)0x80808080, b, 0) != cts_dp4a((int)0x80808080, b, 0))
        atomicExch(bad, 1);
    if (__dp4a(ua, ub, 0u) != cts_dp4a(ua, ub, 0u))
        atomicExch(bad, 1);
    if (__dp4a(0x80808080u, ub, 0u) != cts_dp4a(0x80808080u, ub, 0u))
        atomicExch(bad, 1);
}}
int main() {{
    int* bad = nullptr;
    if (cudaMalloc(&bad, sizeof(*bad)) != cudaSuccess) return 2;
    if (cudaMemset(bad, 0, sizeof(*bad)) != cudaSuccess) return 2;
    compare<<<1, 1>>>(bad);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    int mismatch = 0;
    if (cudaMemcpy(&mismatch, bad, sizeof(mismatch), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    std::printf("mismatch=%d\\n", mismatch);
    cudaFree(bad);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dp4a.cu"
            binary = Path(directory) / "dp4a"
            path.write_text(source, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            compiled = subprocess.run(
                [nvcc, "-arch=native", str(path), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if compiled.returncode != 0:
                self.fail(compiled.stderr)
            ran = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertEqual(ran.stdout.strip(), "mismatch=0")


if __name__ == "__main__":
    unittest.main()


