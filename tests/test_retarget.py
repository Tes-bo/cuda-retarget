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

from cuda_retarget.dialect import BITCAST_HELPERS
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

    def test_hip_and_musa_lower_bitcasts(self) -> None:
        for target in ("hip", "musa"):
            translated = translate_source(BITCAST_SOURCE, target)
            self.assertIn('"__float_as_int"', translated)
            for intrinsic, (name, _helper) in BITCAST_HELPERS.items():
                self.assertNotIn(intrinsic + "(", translated)
                self.assertIn(name + "(", translated)
                self.assertEqual(translated.count(name + "("), 2)
        self.assertEqual(translate_source(BITCAST_SOURCE, "maca"), BITCAST_SOURCE)

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


if __name__ == "__main__":
    unittest.main()
