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
    BYTE_PERM_HELPER,
    HADD_HELPERS,
    MULHI_HELPERS,
    SAD_HELPERS,
    DOUBLE_WORD_HELPERS,
    DP4A_HELPER,
    FUNNEL_HELPERS,
    MATCH_ALL_HELPER,
    MATCH_ANY_HELPER,
    PACKED_ABS_HELPERS,
    PACKED_HELPERS,
    PACKED_MINMAX_HELPERS,
    BIT_HELPERS,
    MUL24_HELPERS,
    ROUND_BITS,
    ROUND_HELPERS,
    ROUND_I32,
    ROUND_I64,
    SATURATE_HELPERS,
    FARITH_ADD,
    FARITH_FMA,
    FARITH_HELPERS,
    FARITH_MUL,
    FARITH_PACK,
    PACKED_AVG_HELPERS,
    PACKED_CMP_HELPERS,
    PACKED_SAD_HELPERS,
    PACKED_SET_HELPERS,
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

    def test_hip_and_musa_lower_packed_arith(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void lanes(unsigned* out) {
    const char* label = "__vadd2";
    unsigned a = 0x7FFF0001u;
    unsigned b = 0x0001FFFFu;
    out[0] = __vadd2(a, b);
    out[1] = __vaddss2(a, b);
    out[2] = __vaddus2(a, b);
    out[3] = __vsubus4(__vadd4(a, b), b);
    out[4] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vadd2"', translated)
            self.assertIn("cts_vadd2(a, b)", translated)
            self.assertIn("cts_vaddss2(a, b)", translated)
            self.assertIn("cts_vaddus2(a, b)", translated)
            self.assertIn("cts_vsubus4(cts_vadd4(a, b), b)", translated)
            self.assertNotIn("__vadd2(", translated)
            self.assertNotIn("__vaddss2(", translated)
            self.assertNotIn("__vadd4(", translated)
            self.assertIn("32767", translated)
            self.assertIn("0xffffu", translated)
            self.assertIn("0xffu", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_packed_minmax(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void pick(unsigned* out) {
    const char* label = "__vmins2";
    unsigned a = 0x80000001u;
    unsigned b = 0x7FFF0000u;
    out[0] = __vmins2(a, b);
    out[1] = __vminu2(a, b);
    out[2] = __vmaxs2(a, b);
    out[3] = __vmaxu4(__vmins4(a, b), b);
    out[4] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vmins2"', translated)
            self.assertIn("cts_vmins2(a, b)", translated)
            self.assertIn("cts_vminu2(a, b)", translated)
            self.assertIn("cts_vmaxs2(a, b)", translated)
            self.assertIn("cts_vmaxu4(cts_vmins4(a, b), b)", translated)
            self.assertNotIn("__vmins2(", translated)
            self.assertNotIn("__vminu2(", translated)
            self.assertIn("static_cast<short>", translated)
            self.assertIn("0xffu", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_packed_abs(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void magnitude(unsigned* out) {
    const char* label = "__vabs2";
    unsigned a = 0x80000002u;
    unsigned b = 0x00000001u;
    out[0] = __vabs2(a);
    out[1] = __vabsss2(a);
    out[2] = __vneg2(a);
    out[3] = __vnegss2(a);
    out[4] = __vnegss4(__vabs4(a));
    out[5] = __vabsdiffs2(a, b);
    out[6] = __vabsdiffu2(a, b);
    out[7] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vabs2"', translated)
            self.assertIn("cts_vabs2(a)", translated)
            self.assertIn("cts_vabsss2(a)", translated)
            self.assertIn("cts_vneg2(a)", translated)
            self.assertIn("cts_vnegss2(a)", translated)
            self.assertIn("cts_vnegss4(cts_vabs4(a))", translated)
            self.assertIn("cts_vabsdiffs2(a, b)", translated)
            self.assertIn("cts_vabsdiffu2(a, b)", translated)
            self.assertNotIn("__vabs2(", translated)
            self.assertNotIn("__vnegss4(", translated)
            self.assertIn("32767", translated)
            self.assertIn("lane == -128", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_byte_perm(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void pick(unsigned* out) {
    const char* label = "__byte_perm";
    unsigned x = 0xA1B2C3D4u;
    unsigned y = 0x1F2E3D4Cu;
    out[0] = __byte_perm(x, y, 0x3210u);
    out[1] = __byte_perm(x, y, 0x32107654u);
    out[2] = __byte_perm(__byte_perm(x, y, 0x89ABu), y, 0x0000u);
    out[3] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__byte_perm"', translated)
            self.assertIn("cts_byte_perm(x, y, 0x3210u)", translated)
            self.assertIn("cts_byte_perm(x, y, 0x32107654u)", translated)
            self.assertIn("cts_byte_perm(cts_byte_perm(x, y, 0x89ABu), y, 0x0000u)", translated)
            self.assertNotIn("__byte_perm(", translated)
            self.assertIn("& 7u", translated)
            self.assertEqual(translated.count("cts_byte_perm("), 5)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_averages(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void average(int* s, unsigned* u) {
    const char* label = "__hadd";
    *s = __hadd(1, 2);
    *s += __rhadd(-3, -2);
    *u = __uhadd(0xffffffffu, 1u);
    *u += __urhadd(1u, 2u);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__hadd"', translated)
            self.assertIn("cts_hadd(1, 2)", translated)
            self.assertIn("cts_rhadd(-3, -2)", translated)
            self.assertIn("cts_uhadd(0xffffffffu, 1u)", translated)
            self.assertIn("cts_urhadd(1u, 2u)", translated)
            self.assertNotIn("__hadd(", translated)
            self.assertNotIn("__uhadd(", translated)
            self.assertIn("(x & y)", translated)
            self.assertIn("(x | y)", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_high_multiplies(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void product(int* s, unsigned* u, long long* w, unsigned long long* z) {
    const char* label = "__mulhi";
    int a = (int)0x80000000;
    *s = __mulhi(__mulhi(a, a), 1);
    *u = __umulhi(0xffffffffu, 0xffffffffu);
    *w = __mul64hi(-3, 5);
    *z = __umul64hi(~0ull, ~0ull);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__mulhi"', translated)
            self.assertIn("cts_mulhi(cts_mulhi(a, a), 1)", translated)
            self.assertIn("cts_umulhi(0xffffffffu, 0xffffffffu)", translated)
            self.assertIn("cts_mul64hi(-3, 5)", translated)
            self.assertIn("cts_umul64hi(~0ull, ~0ull)", translated)
            self.assertNotIn("__mulhi(", translated)
            self.assertNotIn("__mul64hi(", translated)
            self.assertIn(">> 32", translated)
            self.assertLess(
                translated.index("cts_umul64hi(unsigned long long"),
                translated.index("cts_mul64hi(long long"),
            )
        signed_only = """\
#include <cuda_runtime.h>
__global__ void product(long long* w) {
    *w = __mul64hi(-3, 5);
}
"""
        signed = translate_source(signed_only, "hip")
        self.assertIn("cts_umul64hi(unsigned long long", signed)
        self.assertLess(
            signed.index("cts_umul64hi(unsigned long long"),
            signed.index("cts_mul64hi(long long"),
        )
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_absolute_differences(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void distance(unsigned* out) {
    const char* label = "__sad";
    int a = (int)0x80000000;
    out[0] = __usad(__sad(-1, 1, 0u), 0u, 1u);
    out[1] = __sad(a, 0, 0u);
    out[2] = __usad(0xffffffffu, 0u, 0u);
    out[3] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__sad"', translated)
            self.assertIn("cts_usad(cts_sad(-1, 1, 0u), 0u, 1u)", translated)
            self.assertIn("cts_sad(a, 0, 0u)", translated)
            self.assertIn("cts_usad(0xffffffffu, 0u, 0u)", translated)
            self.assertNotIn("__sad(", translated)
            self.assertNotIn("__usad(", translated)
            self.assertIn("static_cast<long long>", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_packed_sad(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void distance(unsigned* out) {
    const char* label = "__vsads2";
    unsigned a = 0xFFFF0001u;
    out[0] = __vsads2(__vsadu2(a, 0u), 0u);
    out[1] = __vsads4(0xFF0180FFu, 0u);
    out[2] = __vsadu4(0xFF0180FFu, 0u);
    out[3] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vsads2"', translated)
            self.assertIn("cts_vsads2(cts_vsadu2(a, 0u), 0u)", translated)
            self.assertIn("cts_vsads4(0xFF0180FFu, 0u)", translated)
            self.assertIn("cts_vsadu4(0xFF0180FFu, 0u)", translated)
            self.assertNotIn("__vsads2(", translated)
            self.assertNotIn("__vsadu4(", translated)
            self.assertIn("unsigned sum = 0;", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_packed_compares(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void compare(unsigned* out) {
    const char* label = "__vcmpgts2";
    unsigned a = 0x1234aba5u;
    unsigned b = 0x1234aba6u;
    unsigned t = 0x7FFF0000u;
    out[0] = __vcmpgts2(__vcmpeq2(a, b), t);
    out[1] = __vcmpgtu4(0x8001807Fu, 0x7F017F00u);
    out[2] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vcmpgts2"', translated)
            self.assertIn("cts_vcmpgts2(cts_vcmpeq2(a, b), t)", translated)
            self.assertIn("cts_vcmpgtu4(0x8001807Fu, 0x7F017F00u)", translated)
            self.assertNotIn("__vcmpgts2(", translated)
            self.assertNotIn("__vcmpeq2(", translated)
            self.assertIn("0xffffu", translated)
            self.assertIn("0xffu", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_lane_predicates(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void compare(unsigned* out) {
    const char* label = "__vsetgts2";
    unsigned a = 0x1234aba5u;
    unsigned b = 0x1234aba6u;
    unsigned t = 0x7FFF0000u;
    out[0] = __vsetgts2(__vseteq2(a, b), t);
    out[1] = __vsetgtu4(0x8001807Fu, 0x7F017F00u);
    out[2] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vsetgts2"', translated)
            self.assertIn("cts_vsetgts2(cts_vseteq2(a, b), t)", translated)
            self.assertIn("cts_vsetgtu4(0x8001807Fu, 0x7F017F00u)", translated)
            self.assertNotIn("__vsetgts2(", translated)
            self.assertNotIn("__vseteq2(", translated)
            self.assertIn("1u <<", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_packed_averages(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void average(unsigned* out) {
    const char* label = "__vavgs2";
    out[0] = __vavgs2(__vhaddu2(0x00010001u, 0x00020002u), 0u);
    out[1] = __vavgu2(0xFFFFFFFFu, 0x00010001u);
    out[2] = label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__vavgs2"', translated)
            self.assertIn("cts_vavgs2(cts_vhaddu2(0x00010001u, 0x00020002u), 0u)", translated)
            self.assertIn("cts_vavgu2(0xFFFFFFFFu, 0x00010001u)", translated)
            self.assertNotIn("__vavgs2(", translated)
            self.assertNotIn("__vhaddu2(", translated)
            self.assertIn("sum < 0", translated)
            self.assertIn("(aa | bb)", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_bit_scans(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void bits(int* s, unsigned* u, unsigned long long* z) {
    const char* label = "__clz";
    *s = __clz(__ffs(0));
    *s += __clzll(0);
    *u = __brev(1u);
    *z = __brevll(1ull);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__clz"', translated)
            self.assertIn("cts_clz(cts_ffs(0))", translated)
            self.assertIn("cts_clzll(0)", translated)
            self.assertIn("cts_brev(1u)", translated)
            self.assertIn("cts_brevll(1ull)", translated)
            self.assertNotIn("__clz(", translated)
            self.assertNotIn("__clzll(", translated)
            self.assertNotIn("__brev(", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_mul24(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void product(int* s, unsigned* u) {
    const char* label = "__mul24";
    *s = __mul24(__mul24((int)0xFF000001, 2), 1);
    *u = __umul24(0xFF800000u, 2u);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__mul24"', translated)
            self.assertIn("cts_mul24(cts_mul24((int)0xFF000001, 2), 1)", translated)
            self.assertIn("cts_umul24(0xFF800000u, 2u)", translated)
            self.assertNotIn("__mul24(", translated)
            self.assertNotIn("__umul24(", translated)
            self.assertIn("0x800000", translated)
            self.assertIn("0xffffffu", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_rounding(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void convert(int* s, float* f, long long* z) {
    const char* label = "__float2int_rn";
    *s = __float2int_rn(2.5f);
    *f = __int2float_ru(16777217);
    *z = __float2ll_rn(*f);
    *f = __ll2float_rd(*z);
    *s = __double2int_rz(1.9);
    *s += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__float2int_rn"', translated)
            self.assertIn("cts_float2int_rn(2.5f)", translated)
            self.assertIn("cts_int2float_ru(16777217)", translated)
            self.assertIn("cts_float2ll_rn(*f)", translated)
            self.assertIn("cts_ll2float_rd(*z)", translated)
            self.assertIn("cts_double2int_rz(1.9)", translated)
            self.assertIn("cts_round_i32", translated)
            self.assertIn("cts_round_u64", translated)
            self.assertIn("0x80000000", translated)
            self.assertNotIn("__float2int_rn(", translated)
            self.assertNotIn("__int2float_ru(", translated)
            self.assertNotIn("__float2ll_rn(", translated)
            self.assertNotIn("__ll2float_rd(", translated)
            self.assertNotIn("__double2int_rz(", translated)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_saturatef(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void clamp(float* f) {
    const char* label = "__saturatef";
    *f = __saturatef(__saturatef(-1.0f));
    *f += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__saturatef"', translated)
            self.assertIn("cts_saturatef(cts_saturatef(-1.0f))", translated)
            self.assertIn("x <= 0.0f", translated)
            self.assertNotIn("__saturatef(", translated)
            self.assertEqual(translated.count("cts_saturatef("), 3)
        self.assertEqual(translate_source(source, "maca"), source)

    def test_hip_and_musa_lower_directed_float(self) -> None:
        source = """\
#include <cuda_runtime.h>
__global__ void round_float(float* f) {
    const char* label = "__fadd_rn __fsub_rd __fmul_ru __fmaf_rz";
    *f = __fadd_rn(*f, 1.0f);
    *f = __fsub_rd(*f, 1.0f);
    *f = __fmul_ru(*f, 2.0f);
    *f = __fmaf_rz(*f, *f, 1.0f);
    *f += label[0];
}
"""
        for target in ("hip", "musa"):
            translated = translate_source(source, target)
            self.assertIn('"__fadd_rn __fsub_rd __fmul_ru __fmaf_rz"', translated)
            self.assertIn("cts_fadd_rn(*f, 1.0f)", translated)
            self.assertIn("cts_fsub_rd(*f, 1.0f)", translated)
            self.assertIn("cts_fmul_ru(*f, 2.0f)", translated)
            self.assertIn("cts_fmaf_rz(*f, *f, 1.0f)", translated)
            self.assertIn("cts_fpack", translated)
            self.assertIn("cts_fadd(", translated)
            self.assertIn("cts_ffma(", translated)
            self.assertNotIn("__fadd_rn(", translated)
            self.assertNotIn("__fsub_rd(", translated)
            self.assertNotIn("__fmul_ru(", translated)
            self.assertNotIn("__fmaf_rz(", translated)
        only = """\
#include <cuda_runtime.h>
__global__ void round_float(float* f) { *f = __fadd_rn(*f, 1.0f); }
"""
        hip = translate_source(only, "hip")
        self.assertIn("cts_fpack", hip)
        self.assertIn("cts_fadd(", hip)
        self.assertNotIn("cts_fmul(", hip)
        self.assertNotIn("cts_ffma(", hip)
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

    def test_hip_and_musa_lower_the_census_packed_add(self) -> None:
        path = CENSUS / "packed_add" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_add is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vadd2(a2, b2)", translated)
            self.assertIn("cts_vaddss2(a2, b2)", translated)
            self.assertIn("cts_vaddus2(a2, b2)", translated)
            self.assertIn("cts_vsubss4(s4, t4)", translated)
            self.assertNotIn("__vadd2(", translated)
            self.assertEqual(translated.count("cts_vadd2("), 2)
        self.assertIn("__vadd2", maca)
        self.assertNotIn("cts_vadd2", maca)

    def test_hip_and_musa_lower_the_census_packed_minmax(self) -> None:
        path = CENSUS / "packed_minmax" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_minmax is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vmins2(a2, b2)", translated)
            self.assertIn("cts_vminu2(a2, b2)", translated)
            self.assertIn("cts_vmaxu4(a4, b4)", translated)
            self.assertNotIn("__vmins2(", translated)
            self.assertEqual(translated.count("cts_vmins2("), 2)
        self.assertIn("__vmins2", maca)
        self.assertNotIn("cts_vmins2", maca)

    def test_hip_and_musa_lower_the_census_packed_abs(self) -> None:
        path = CENSUS / "packed_abs" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_abs is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vabs2(a2)", translated)
            self.assertIn("cts_vabsss2(a2)", translated)
            self.assertIn("cts_vnegss4(a4)", translated)
            self.assertIn("cts_vabsdiffu4(x4, y4)", translated)
            self.assertNotIn("__vabs2(", translated)
            self.assertEqual(translated.count("cts_vabs2("), 2)
        self.assertIn("__vabs2", maca)
        self.assertNotIn("cts_vabs2", maca)

    def test_hip_and_musa_lower_the_census_byte_perm(self) -> None:
        path = CENSUS / "byte_perm" / "main.cu"
        if not path.is_file():
            self.skipTest("byte_perm is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_byte_perm(x, y, 0x3210u)", translated)
            self.assertIn("cts_byte_perm(x, y, 0x7654u)", translated)
            self.assertIn("cts_byte_perm(x, y, 0x32107654u)", translated)
            self.assertNotIn("__byte_perm(", translated)
            self.assertEqual(translated.count("cts_byte_perm("), 6)
        self.assertIn("__byte_perm", maca)
        self.assertNotIn("cts_byte_perm", maca)

    def test_hip_and_musa_lower_the_census_average(self) -> None:
        path = CENSUS / "average" / "main.cu"
        if not path.is_file():
            self.skipTest("average is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_hadd(1, 2)", translated)
            self.assertIn("cts_rhadd(1, 2)", translated)
            self.assertIn("cts_hadd(-3, -2)", translated)
            self.assertIn("cts_uhadd(0xffffffffu, 1u)", translated)
            self.assertNotIn("__hadd(", translated)
            self.assertEqual(translated.count("cts_hadd("), 4)
        self.assertIn("__hadd", maca)
        self.assertNotIn("cts_hadd", maca)

    def test_hip_and_musa_lower_the_census_mulhi(self) -> None:
        path = CENSUS / "mulhi" / "main.cu"
        if not path.is_file():
            self.skipTest("mulhi is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_mulhi(a, 1)", translated)
            self.assertIn("cts_mulhi(a, -1)", translated)
            self.assertIn("cts_mulhi(a, a)", translated)
            self.assertIn("cts_umulhi(0xffffffffu, 0xffffffffu)", translated)
            self.assertIn("cts_mul64hi(-3, 5)", translated)
            self.assertIn("cts_mul64hi(m, m)", translated)
            self.assertIn("cts_umul64hi(~0ull, ~0ull)", translated)
            self.assertNotIn("__mulhi(", translated)
            self.assertNotIn("__mul64hi(", translated)
            self.assertEqual(translated.count("cts_mulhi("), 4)
        self.assertIn("__mulhi", maca)
        self.assertNotIn("cts_mulhi", maca)

    def test_hip_and_musa_lower_the_census_sad(self) -> None:
        path = CENSUS / "sad" / "main.cu"
        if not path.is_file():
            self.skipTest("sad is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_sad(-1, 1, 0u)", translated)
            self.assertIn("cts_sad(a, 0, 0u)", translated)
            self.assertIn("cts_sad(a, 1, 0u)", translated)
            self.assertIn("cts_sad(1, 0, 0xffffffffu)", translated)
            self.assertIn("cts_usad(0xffffffffu, 0u, 0u)", translated)
            self.assertIn("cts_usad(0u, 1u, 0xffffffffu)", translated)
            self.assertNotIn("__sad(", translated)
            self.assertNotIn("__usad(", translated)
            self.assertEqual(translated.count("cts_sad("), 5)
        self.assertIn("__sad", maca)
        self.assertNotIn("cts_sad", maca)

    def test_hip_and_musa_lower_the_census_packed_sad(self) -> None:
        path = CENSUS / "packed_sad" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_sad is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vsads2(a2, 0u)", translated)
            self.assertIn("cts_vsadu2(a2, 0u)", translated)
            self.assertIn("cts_vsads2(m2, 0u)", translated)
            self.assertIn("cts_vsads2(left, right)", translated)
            self.assertIn("cts_vsads4(a4, 0u)", translated)
            self.assertIn("cts_vsadu4(a4, 0u)", translated)
            self.assertNotIn("__vsads2(", translated)
            self.assertNotIn("__vsadu4(", translated)
            self.assertEqual(translated.count("cts_vsads2("), 4)
        self.assertIn("__vsads2", maca)
        self.assertNotIn("cts_vsads2", maca)

    def test_hip_and_musa_lower_the_census_packed_cmp(self) -> None:
        path = CENSUS / "packed_cmp" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_cmp is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vcmpeq2(0x1234aba5u, 0x1234aba6u)", translated)
            self.assertIn("cts_vcmpne2(0x1234aba5u, 0x1234aba6u)", translated)
            self.assertIn("cts_vcmpgts2(s, t)", translated)
            self.assertIn("cts_vcmpgtu2(s, t)", translated)
            self.assertIn("cts_vcmplts2(s, t)", translated)
            self.assertIn("cts_vcmpgts4(c, d)", translated)
            self.assertNotIn("__vcmpeq2(", translated)
            self.assertNotIn("__vcmpgts4(", translated)
            self.assertEqual(translated.count("cts_vcmpgts2("), 2)
        self.assertIn("__vcmpgts2", maca)
        self.assertNotIn("cts_vcmpgts2", maca)

    def test_hip_and_musa_lower_the_census_packed_set(self) -> None:
        path = CENSUS / "packed_set" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_set is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vseteq2(0x1234aba5u, 0x1234aba6u)", translated)
            self.assertIn("cts_vsetne2(0x1234aba5u, 0x1234aba6u)", translated)
            self.assertIn("cts_vsetgts2(s, t)", translated)
            self.assertIn("cts_vsetgtu2(s, t)", translated)
            self.assertIn("cts_vsetgts4(c, d)", translated)
            self.assertIn("cts_vsetgtu4(c, d)", translated)
            self.assertNotIn("__vseteq2(", translated)
            self.assertEqual(translated.count("cts_vsetgts2("), 2)
        self.assertIn("__vsetgts2", maca)
        self.assertNotIn("cts_vsetgts2", maca)

    def test_hip_and_musa_lower_the_census_packed_avg(self) -> None:
        path = CENSUS / "packed_avg" / "main.cu"
        if not path.is_file():
            self.skipTest("packed_avg is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_vavgs2(0x00010001u, 0x00020002u)", translated)
            self.assertIn("cts_vavgs2(0xFFFDFFFEu, 0xFFFEFFFDu)", translated)
            self.assertIn("cts_vavgs2(0x80008000u, 0x80008000u)", translated)
            self.assertIn("cts_vavgu2(0x00010001u, 0x00020002u)", translated)
            self.assertIn("cts_vhaddu2(0xFFFFFFFFu, 0x00010001u)", translated)
            self.assertNotIn("__vavgs2(", translated)
            self.assertEqual(translated.count("cts_vavgs2("), 4)
        self.assertIn("__vavgs2", maca)
        self.assertNotIn("cts_vavgs2", maca)

    def test_hip_and_musa_lower_the_census_bit_scan(self) -> None:
        path = CENSUS / "bit_scan" / "main.cu"
        if not path.is_file():
            self.skipTest("bit_scan is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_clz(0)", translated)
            self.assertIn("cts_clz((int)0x80000000)", translated)
            self.assertIn("cts_ffs(0)", translated)
            self.assertIn("cts_ffs((int)0x80000000)", translated)
            self.assertIn("cts_clzll(0)", translated)
            self.assertIn("cts_popc(0xffffffffu)", translated)
            self.assertIn("cts_brev(1u)", translated)
            self.assertIn("cts_brevll(1ull)", translated)
            self.assertNotIn("__clz(", translated)
            self.assertNotIn("__clzll(", translated)
            self.assertEqual(translated.count("cts_clz("), 3)
        self.assertIn("__clz", maca)
        self.assertNotIn("cts_clz", maca)

    def test_hip_and_musa_lower_the_census_mul24(self) -> None:
        path = CENSUS / "mul24" / "main.cu"
        if not path.is_file():
            self.skipTest("mul24 is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_mul24((int)0xFF000001, 2)", translated)
            self.assertIn("cts_mul24((int)0x00800000, 1)", translated)
            self.assertIn("cts_mul24((int)0x00FFFFFF, (int)0x00FFFFFF)", translated)
            self.assertIn("cts_mul24((int)0x00800000, (int)0x00800000)", translated)
            self.assertIn("cts_umul24(0x00FFFFFFu, 0x00FFFFFFu)", translated)
            self.assertIn("cts_umul24(0xFF800000u, 2u)", translated)
            self.assertNotIn("__mul24(", translated)
            self.assertEqual(translated.count("cts_mul24("), 5)
        self.assertIn("__mul24", maca)
        self.assertNotIn("cts_mul24", maca)

    def test_hip_and_musa_lower_the_census_round_cvt(self) -> None:
        path = CENSUS / "round_cvt" / "main.cu"
        if not path.is_file():
            self.skipTest("round_cvt is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_float2int_rn(2.5f)", translated)
            self.assertIn("cts_float2int_ru(2.5f)", translated)
            self.assertIn("cts_float2int_rd(-2.5f)", translated)
            self.assertIn("cts_int2float_ru(16777217)", translated)
            self.assertIn("cts_int2float_rd(-16777219)", translated)
            self.assertIn("cts_uint2float_rd(0xffffffffu)", translated)
            self.assertNotIn("__float2int_rn(", translated)
            self.assertNotIn("__int2float_ru(", translated)
            self.assertEqual(translated.count("cts_float2int_rn("), 5)
            self.assertEqual(translated.count("cts_int2float_rd("), 3)
        self.assertIn("__float2int_rn", maca)
        self.assertNotIn("cts_float2int_rn", maca)

    def test_hip_and_musa_lower_the_census_round_wide(self) -> None:
        path = CENSUS / "round_wide" / "main.cu"
        if not path.is_file():
            self.skipTest("round_wide is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_float2ll_rn(nan.as_float)", translated)
            self.assertIn("cts_float2ll_rn(1.0e20f)", translated)
            self.assertIn("cts_ll2float_rn((1LL << 53) + 1)", translated)
            self.assertIn("cts_ll2float_ru((1LL << 53) + 1)", translated)
            self.assertIn("cts_ull2float_rd(0xffffffffffffffffull)", translated)
            self.assertIn("cts_double2int_rz(dnan.as_double)", translated)
            self.assertIn("cts_double2ull_rz(1.0e20)", translated)
            self.assertIn("cts_round_u64", translated)
            self.assertNotIn("__float2ll_rn(", translated)
            self.assertNotIn("__double2int_rz(", translated)
            self.assertEqual(translated.count("cts_float2ll_rn("), 3)
        self.assertIn("__float2ll_rn", maca)
        self.assertNotIn("cts_float2ll_rn", maca)

    def test_hip_and_musa_lower_the_census_saturatef(self) -> None:
        path = CENSUS / "saturatef" / "main.cu"
        if not path.is_file():
            self.skipTest("saturatef is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_saturatef(-1.0f)", translated)
            self.assertIn("cts_saturatef(2.0f)", translated)
            self.assertIn("cts_saturatef(0.5f)", translated)
            self.assertIn("cts_saturatef(-0.0f)", translated)
            self.assertIn("cts_saturatef(nan.as_float)", translated)
            self.assertNotIn("__saturatef(", translated)
            self.assertEqual(translated.count("cts_saturatef("), 6)
        self.assertIn("__saturatef", maca)
        self.assertNotIn("cts_saturatef", maca)

    def test_hip_and_musa_lower_the_census_farith(self) -> None:
        path = CENSUS / "farith" / "main.cu"
        if not path.is_file():
            self.skipTest("farith is not in the census corpus")
        hip = translate_path(path, "hip")
        musa = translate_path(path, "musa")
        maca = translate_path(path, "maca")
        for translated in (hip, musa):
            self.assertIn("cts_fadd_rn(1.0f, tiny.as_float)", translated)
            self.assertIn("cts_fadd_ru(1.0f, tiny.as_float)", translated)
            self.assertIn("cts_fadd_rd(0.0f, 0.0f)", translated)
            self.assertIn("cts_fadd_rd(0.0f, -0.0f)", translated)
            self.assertIn("cts_fsub_rd(1.0f, tiny.as_float)", translated)
            self.assertIn("cts_fmul_rn(-0.0f, 0.0f)", translated)
            self.assertIn("cts_fmaf_ru(1.0f, tiny.as_float, 1.0f)", translated)
            self.assertIn("cts_fmaf_rn(wide.as_float, wide.as_float, cts_uint_bits(0xbf800002u))", translated)
            self.assertIn("cts_fmaf_rn(0.0f, cts_uint_bits(0x7f800000u), 1.0f)", translated)
            self.assertIn("cts_fpack", translated)
            self.assertNotIn("__fadd_rn(", translated)
            self.assertNotIn("__fmaf_rn(", translated)
            self.assertEqual(translated.count("cts_fadd_rn("), 5)
            self.assertEqual(translated.count("cts_fmaf_rn("), 3)
        self.assertIn("__fadd_rn", maca)
        self.assertNotIn("cts_fadd_rn", maca)

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

    def test_packed_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a2 = 0x7FFF0001u;
    unsigned b2 = 0x0001FFFFu;
    unsigned s2 = 0x80000000u;
    unsigned t2 = 0x00010001u;
    unsigned a4 = 0x7F0180FFu;
    unsigned b4 = 0x01010101u;
    unsigned s4 = 0x80018000u;
    unsigned t4 = 0x01010101u;
    if (__vadd2(a2, b2) != cts_vadd2(a2, b2)) atomicExch(bad, 1);
    if (__vaddss2(a2, b2) != cts_vaddss2(a2, b2)) atomicExch(bad, 1);
    if (__vaddus2(a2, b2) != cts_vaddus2(a2, b2)) atomicExch(bad, 1);
    if (__vsub2(s2, t2) != cts_vsub2(s2, t2)) atomicExch(bad, 1);
    if (__vsubss2(s2, t2) != cts_vsubss2(s2, t2)) atomicExch(bad, 1);
    if (__vsubus2(s2, t2) != cts_vsubus2(s2, t2)) atomicExch(bad, 1);
    if (__vadd4(a4, b4) != cts_vadd4(a4, b4)) atomicExch(bad, 1);
    if (__vaddss4(a4, b4) != cts_vaddss4(a4, b4)) atomicExch(bad, 1);
    if (__vaddus4(a4, b4) != cts_vaddus4(a4, b4)) atomicExch(bad, 1);
    if (__vsub4(s4, t4) != cts_vsub4(s4, t4)) atomicExch(bad, 1);
    if (__vsubss4(s4, t4) != cts_vsubss4(s4, t4)) atomicExch(bad, 1);
    if (__vsubus4(s4, t4) != cts_vsubus4(s4, t4)) atomicExch(bad, 1);
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
            path = Path(directory) / "packed.cu"
            binary = Path(directory) / "packed"
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

    def test_packed_minmax_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_MINMAX_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a2 = 0x80000001u;
    unsigned b2 = 0x7FFF0000u;
    unsigned a4 = 0x8001807Fu;
    unsigned b4 = 0x7F017F00u;
    unsigned eq = 0x00010001u;
    if (__vmins2(a2, b2) != cts_vmins2(a2, b2)) atomicExch(bad, 1);
    if (__vminu2(a2, b2) != cts_vminu2(a2, b2)) atomicExch(bad, 1);
    if (__vmaxs2(a2, b2) != cts_vmaxs2(a2, b2)) atomicExch(bad, 1);
    if (__vmaxu2(a2, b2) != cts_vmaxu2(a2, b2)) atomicExch(bad, 1);
    if (__vmins4(a4, b4) != cts_vmins4(a4, b4)) atomicExch(bad, 1);
    if (__vminu4(a4, b4) != cts_vminu4(a4, b4)) atomicExch(bad, 1);
    if (__vmaxs4(a4, b4) != cts_vmaxs4(a4, b4)) atomicExch(bad, 1);
    if (__vmaxu4(a4, b4) != cts_vmaxu4(a4, b4)) atomicExch(bad, 1);
    if (__vmins2(eq, eq) != cts_vmins2(eq, eq)) atomicExch(bad, 1);
    if (__vmaxu4(eq, eq) != cts_vmaxu4(eq, eq)) atomicExch(bad, 1);
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
            path = Path(directory) / "minmax.cu"
            binary = Path(directory) / "minmax"
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

    def test_packed_abs_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper, _arity in PACKED_ABS_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a2 = 0x80000002u;
    unsigned a4 = 0x80017F80u;
    unsigned x2 = 0x8000FFFFu;
    unsigned y2 = 0x00000001u;
    unsigned x4 = 0x01800080u;
    unsigned y4 = 0x00010100u;
    if (__vabs2(a2) != cts_vabs2(a2)) atomicExch(bad, 1);
    if (__vabsss2(a2) != cts_vabsss2(a2)) atomicExch(bad, 1);
    if (__vneg2(a2) != cts_vneg2(a2)) atomicExch(bad, 1);
    if (__vnegss2(a2) != cts_vnegss2(a2)) atomicExch(bad, 1);
    if (__vnegss2(0x80008000u) != cts_vnegss2(0x80008000u)) atomicExch(bad, 1);
    if (__vabs4(a4) != cts_vabs4(a4)) atomicExch(bad, 1);
    if (__vabsss4(a4) != cts_vabsss4(a4)) atomicExch(bad, 1);
    if (__vneg4(a4) != cts_vneg4(a4)) atomicExch(bad, 1);
    if (__vnegss4(a4) != cts_vnegss4(a4)) atomicExch(bad, 1);
    if (__vabsdiffs2(x2, y2) != cts_vabsdiffs2(x2, y2)) atomicExch(bad, 1);
    if (__vabsdiffu2(x2, y2) != cts_vabsdiffu2(x2, y2)) atomicExch(bad, 1);
    if (__vabsdiffs4(x4, y4) != cts_vabsdiffs4(x4, y4)) atomicExch(bad, 1);
    if (__vabsdiffu4(x4, y4) != cts_vabsdiffu4(x4, y4)) atomicExch(bad, 1);
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
            path = Path(directory) / "abs.cu"
            binary = Path(directory) / "abs"
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

    def test_byte_perm_helper_agrees_with_the_intrinsic(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = BYTE_PERM_HELPER.strip("\n")
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned x = 0xA1B2C3D4u;
    unsigned y = 0x1F2E3D4Cu;
    unsigned selectors[5] = {{0x3210u, 0x7654u, 0x0401u, 0x89ABu, 0x32107654u}};
    for (int i = 0; i < 5; ++i) {{
        if (__byte_perm(x, y, selectors[i]) != cts_byte_perm(x, y, selectors[i]))
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
            path = Path(directory) / "perm.cu"
            binary = Path(directory) / "perm"
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

    def test_average_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in HADD_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    if (__hadd(1, 2) != cts_hadd(1, 2)) atomicExch(bad, 1);
    if (__rhadd(1, 2) != cts_rhadd(1, 2)) atomicExch(bad, 1);
    if (__hadd(-3, -2) != cts_hadd(-3, -2)) atomicExch(bad, 1);
    if (__rhadd(-3, -2) != cts_rhadd(-3, -2)) atomicExch(bad, 1);
    if (__hadd((int)0x80000000, (int)0x80000000) != cts_hadd((int)0x80000000, (int)0x80000000))
        atomicExch(bad, 1);
    if (__rhadd(2147483647, 2147483647) != cts_rhadd(2147483647, 2147483647))
        atomicExch(bad, 1);
    if (__uhadd(0xffffffffu, 1u) != cts_uhadd(0xffffffffu, 1u)) atomicExch(bad, 1);
    if (__urhadd(1u, 2u) != cts_urhadd(1u, 2u)) atomicExch(bad, 1);
    if (__uhadd(0xffffffffu, 0xffffffffu) != cts_uhadd(0xffffffffu, 0xffffffffu))
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
            path = Path(directory) / "avg.cu"
            binary = Path(directory) / "avg"
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

    def test_mulhi_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in MULHI_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    int a = (int)0x80000000;
    long long m = (long long)0x8000000000000000ull;
    if (__mulhi(a, 1) != cts_mulhi(a, 1)) atomicExch(bad, 1);
    if (__mulhi(a, -1) != cts_mulhi(a, -1)) atomicExch(bad, 1);
    if (__mulhi(a, a) != cts_mulhi(a, a)) atomicExch(bad, 1);
    if (__mulhi(100000, 100000) != cts_mulhi(100000, 100000)) atomicExch(bad, 1);
    if (__umulhi(0xffffffffu, 0xffffffffu) != cts_umulhi(0xffffffffu, 0xffffffffu))
        atomicExch(bad, 1);
    if (__umulhi(0x80000000u, 2u) != cts_umulhi(0x80000000u, 2u)) atomicExch(bad, 1);
    if (__mul64hi(m, 1) != cts_mul64hi(m, 1)) atomicExch(bad, 1);
    if (__mul64hi(m, -1) != cts_mul64hi(m, -1)) atomicExch(bad, 1);
    if (__mul64hi(m, m) != cts_mul64hi(m, m)) atomicExch(bad, 1);
    if (__mul64hi(-3, 5) != cts_mul64hi(-3, 5)) atomicExch(bad, 1);
    if (__umul64hi(~0ull, ~0ull) != cts_umul64hi(~0ull, ~0ull)) atomicExch(bad, 1);
    if (__umul64hi(~0ull, 2ull) != cts_umul64hi(~0ull, 2ull)) atomicExch(bad, 1);
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
            path = Path(directory) / "mulhi.cu"
            binary = Path(directory) / "mulhi"
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

    def test_sad_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in SAD_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    int a = (int)0x80000000;
    if (__sad(-1, 1, 0u) != cts_sad(-1, 1, 0u)) atomicExch(bad, 1);
    if (__sad(a, 0, 0u) != cts_sad(a, 0, 0u)) atomicExch(bad, 1);
    if (__sad(a, 1, 0u) != cts_sad(a, 1, 0u)) atomicExch(bad, 1);
    if (__sad(a, 2147483647, 0u) != cts_sad(a, 2147483647, 0u)) atomicExch(bad, 1);
    if (__sad(a, a, 0u) != cts_sad(a, a, 0u)) atomicExch(bad, 1);
    if (__sad(1, 0, 0xffffffffu) != cts_sad(1, 0, 0xffffffffu)) atomicExch(bad, 1);
    if (__sad(a, 2147483647, 1u) != cts_sad(a, 2147483647, 1u)) atomicExch(bad, 1);
    if (__usad(0xffffffffu, 0u, 0u) != cts_usad(0xffffffffu, 0u, 0u)) atomicExch(bad, 1);
    if (__usad(0u, 1u, 0xffffffffu) != cts_usad(0u, 1u, 0xffffffffu)) atomicExch(bad, 1);
    if (__usad(0xffffffffu, 0u, 1u) != cts_usad(0xffffffffu, 0u, 1u)) atomicExch(bad, 1);
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
            path = Path(directory) / "sad.cu"
            binary = Path(directory) / "sad"
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

    def test_packed_sad_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_SAD_HELPERS.values())
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a2 = 0xFFFF0001u;
    unsigned m2 = 0x80008000u;
    unsigned left = 0x00020001u;
    unsigned right = 0x00010003u;
    unsigned a4 = 0xFF0180FFu;
    unsigned m4 = 0x80808080u;
    if (__vsads2(a2, 0u) != cts_vsads2(a2, 0u)) atomicExch(bad, 1);
    if (__vsadu2(a2, 0u) != cts_vsadu2(a2, 0u)) atomicExch(bad, 1);
    if (__vsads2(m2, 0u) != cts_vsads2(m2, 0u)) atomicExch(bad, 1);
    if (__vsadu2(m2, 0u) != cts_vsadu2(m2, 0u)) atomicExch(bad, 1);
    if (__vsads2(left, right) != cts_vsads2(left, right)) atomicExch(bad, 1);
    if (__vsadu2(left, right) != cts_vsadu2(left, right)) atomicExch(bad, 1);
    if (__vsads4(a4, 0u) != cts_vsads4(a4, 0u)) atomicExch(bad, 1);
    if (__vsadu4(a4, 0u) != cts_vsadu4(a4, 0u)) atomicExch(bad, 1);
    if (__vsads4(m4, 0u) != cts_vsads4(m4, 0u)) atomicExch(bad, 1);
    if (__vsadu4(m4, 0u) != cts_vsadu4(m4, 0u)) atomicExch(bad, 1);
    if (__vsads2(0x8000FFFFu, 0x7FFF0000u) != cts_vsads2(0x8000FFFFu, 0x7FFF0000u))
        atomicExch(bad, 1);
    if (__vsadu2(0x8000FFFFu, 0x7FFF0000u) != cts_vsadu2(0x8000FFFFu, 0x7FFF0000u))
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
            path = Path(directory) / "psad.cu"
            binary = Path(directory) / "psad"
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

    def test_packed_cmp_helpers_agree_with_the_intrinsics(self) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_CMP_HELPERS.values())
        checks = []
        for intrinsic, (name, _helper) in PACKED_CMP_HELPERS.items():
            for left, right in (("a", "b"), ("s", "t"), ("c", "d"), ("s", "s")):
                checks.append(f"if ({intrinsic}({left}, {right}) != {name}({left}, {right})) atomicExch(bad, 1);")
        body = "\n    ".join(checks)
        source = f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a = 0x1234aba5u;
    unsigned b = 0x1234aba6u;
    unsigned s = 0x80000001u;
    unsigned t = 0x7FFF0000u;
    unsigned c = 0x8001807Fu;
    unsigned d = 0x7F017F00u;
    {body}
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
            path = Path(directory) / "pcmp.cu"
            binary = Path(directory) / "pcmp"
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

    def _nvcc_mismatch(self, name: str, source: str) -> None:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            self.skipTest("nvcc is not installed")
        device = _idle_gpu()
        if device is None:
            self.skipTest("every GPU has a compute process")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{name}.cu"
            binary = Path(directory) / name
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

    def test_lane_predicate_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_SET_HELPERS.values())
        checks = []
        for intrinsic, (name, _helper) in PACKED_SET_HELPERS.items():
            for left, right in (("a", "b"), ("s", "t"), ("c", "d"), ("s", "s")):
                checks.append(
                    f"if ({intrinsic}({left}, {right}) != {name}({left}, {right})) atomicExch(bad, 1);"
                )
        body = "\n    ".join(checks)
        self._nvcc_mismatch(
            "pset",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned a = 0x1234aba5u;
    unsigned b = 0x1234aba6u;
    unsigned s = 0x80000001u;
    unsigned t = 0x7FFF0000u;
    unsigned c = 0x8001807Fu;
    unsigned d = 0x7F017F00u;
    {body}
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
""",
        )

    def test_packed_average_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(helper.strip("\n") for _name, helper in PACKED_AVG_HELPERS.values())
        self._nvcc_mismatch(
            "pavg",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    if (__vavgs2(0x00010001u, 0x00020002u) != cts_vavgs2(0x00010001u, 0x00020002u)) atomicExch(bad, 1);
    if (__vavgs2(0xFFFDFFFEu, 0xFFFEFFFDu) != cts_vavgs2(0xFFFDFFFEu, 0xFFFEFFFDu)) atomicExch(bad, 1);
    if (__vavgs2(0x80008000u, 0x80008000u) != cts_vavgs2(0x80008000u, 0x80008000u)) atomicExch(bad, 1);
    if (__vavgs2(0x80007FFFu, 0x7FFF8000u) != cts_vavgs2(0x80007FFFu, 0x7FFF8000u)) atomicExch(bad, 1);
    if (__vavgu2(0x00010001u, 0x00020002u) != cts_vavgu2(0x00010001u, 0x00020002u)) atomicExch(bad, 1);
    if (__vhaddu2(0x00010001u, 0x00020002u) != cts_vhaddu2(0x00010001u, 0x00020002u)) atomicExch(bad, 1);
    if (__vhaddu2(0xFFFFFFFFu, 0x00010001u) != cts_vhaddu2(0xFFFFFFFFu, 0x00010001u)) atomicExch(bad, 1);
    if (__vavgs4(0xFEFDFEFDu, 0xFDFEFDFEu) != cts_vavgs4(0xFEFDFEFDu, 0xFDFEFDFEu)) atomicExch(bad, 1);
    if (__vavgs4(0x80808080u, 0x80808080u) != cts_vavgs4(0x80808080u, 0x80808080u)) atomicExch(bad, 1);
    if (__vavgu4(0x01010101u, 0x02020202u) != cts_vavgu4(0x01010101u, 0x02020202u)) atomicExch(bad, 1);
    if (__vhaddu4(0xFFFFFFFFu, 0x01010101u) != cts_vhaddu4(0xFFFFFFFFu, 0x01010101u)) atomicExch(bad, 1);
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
""",
        )

    def test_bit_scan_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(helper.strip("\n") for _name, helper in BIT_HELPERS.values())
        self._nvcc_mismatch(
            "bits",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    long long m = (long long)0x8000000000000000ull;
    if (__clz(0) != cts_clz(0)) atomicExch(bad, 1);
    if (__clz(1) != cts_clz(1)) atomicExch(bad, 1);
    if (__clz((int)0x80000000) != cts_clz((int)0x80000000)) atomicExch(bad, 1);
    if (__clz(-1) != cts_clz(-1)) atomicExch(bad, 1);
    if (__clzll(0) != cts_clzll(0)) atomicExch(bad, 1);
    if (__clzll(1) != cts_clzll(1)) atomicExch(bad, 1);
    if (__clzll(m) != cts_clzll(m)) atomicExch(bad, 1);
    if (__ffs(0) != cts_ffs(0)) atomicExch(bad, 1);
    if (__ffs(1) != cts_ffs(1)) atomicExch(bad, 1);
    if (__ffs((int)0x80000000) != cts_ffs((int)0x80000000)) atomicExch(bad, 1);
    if (__ffs(-1) != cts_ffs(-1)) atomicExch(bad, 1);
    if (__ffsll(0) != cts_ffsll(0)) atomicExch(bad, 1);
    if (__ffsll(m) != cts_ffsll(m)) atomicExch(bad, 1);
    if (__popc(0xffffffffu) != cts_popc(0xffffffffu)) atomicExch(bad, 1);
    if (__popc(0) != cts_popc(0)) atomicExch(bad, 1);
    if (__popcll(~0ull) != cts_popcll(~0ull)) atomicExch(bad, 1);
    if (__brev(1u) != cts_brev(1u)) atomicExch(bad, 1);
    if (__brev(0x80000000u) != cts_brev(0x80000000u)) atomicExch(bad, 1);
    if (__brevll(1ull) != cts_brevll(1ull)) atomicExch(bad, 1);
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
""",
        )

    def test_mul24_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(helper.strip("\n") for _name, helper in MUL24_HELPERS.values())
        self._nvcc_mismatch(
            "m24",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    if (__mul24((int)0xFF000001, 2) != cts_mul24((int)0xFF000001, 2)) atomicExch(bad, 1);
    if (__mul24((int)0x00800000, 1) != cts_mul24((int)0x00800000, 1)) atomicExch(bad, 1);
    if (__mul24((int)0x00FFFFFF, (int)0x00FFFFFF) != cts_mul24((int)0x00FFFFFF, (int)0x00FFFFFF))
        atomicExch(bad, 1);
    if (__mul24((int)0x00800000, (int)0x00800000) != cts_mul24((int)0x00800000, (int)0x00800000))
        atomicExch(bad, 1);
    if (__umul24(0x00FFFFFFu, 0x00FFFFFFu) != cts_umul24(0x00FFFFFFu, 0x00FFFFFFu)) atomicExch(bad, 1);
    if (__umul24(0xFF800000u, 2u) != cts_umul24(0xFF800000u, 2u)) atomicExch(bad, 1);
    if (__umul24(0x01000000u, 3u) != cts_umul24(0x01000000u, 3u)) atomicExch(bad, 1);
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
""",
        )

    def test_rounding_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(
            [
                ROUND_BITS.strip("\n"),
                ROUND_I32.strip("\n"),
                ROUND_I64.strip("\n"),
                *(helper.strip("\n") for _name, helper in ROUND_HELPERS.values()),
            ]
        )
        self._nvcc_mismatch(
            "round",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    union {{ float f; unsigned u; }} nanf;
    nanf.u = 0x7fc00000u;
    union {{ double d; unsigned long long u; }} nand;
    nand.u = 0x7ff8000000000000ull;
    float two63 = __uint_as_float(0x5f000000u);
    if (__float2int_rn(2.5f) != cts_float2int_rn(2.5f)) atomicExch(bad, 1);
    if (__float2int_ru(2.5f) != cts_float2int_ru(2.5f)) atomicExch(bad, 1);
    if (__float2int_rd(-2.5f) != cts_float2int_rd(-2.5f)) atomicExch(bad, 1);
    if (__float2int_rn(1.0e20f) != cts_float2int_rn(1.0e20f)) atomicExch(bad, 1);
    if (__float2int_rn(-1.0e20f) != cts_float2int_rn(-1.0e20f)) atomicExch(bad, 1);
    if (__float2int_rn(nanf.f) != cts_float2int_rn(nanf.f)) atomicExch(bad, 1);
    if (__float2int_rn(0.5f) != cts_float2int_rn(0.5f)) atomicExch(bad, 1);
    if (__float2int_rn(-0.5f) != cts_float2int_rn(-0.5f)) atomicExch(bad, 1);
    if (__float2uint_rn(-1.2f) != cts_float2uint_rn(-1.2f)) atomicExch(bad, 1);
    if (__float2uint_rz(1.0e20f) != cts_float2uint_rz(1.0e20f)) atomicExch(bad, 1);
    if (__float2uint_ru(0.2f) != cts_float2uint_ru(0.2f)) atomicExch(bad, 1);
    if (__float2uint_rn(nanf.f) != cts_float2uint_rn(nanf.f)) atomicExch(bad, 1);
    if (__float_as_int(__int2float_rn(16777217)) != __float_as_int(cts_int2float_rn(16777217)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_ru(16777217)) != __float_as_int(cts_int2float_ru(16777217)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_rz(16777219)) != __float_as_int(cts_int2float_rz(16777219)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_rd(-16777219)) != __float_as_int(cts_int2float_rd(-16777219)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_ru(-16777219)) != __float_as_int(cts_int2float_ru(-16777219)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_rn(2147483647)) != __float_as_int(cts_int2float_rn(2147483647)))
        atomicExch(bad, 1);
    if (__float_as_int(__int2float_rd(2147483647)) != __float_as_int(cts_int2float_rd(2147483647)))
        atomicExch(bad, 1);
    if (__float_as_uint(__uint2float_rn(0xffffffffu)) != __float_as_uint(cts_uint2float_rn(0xffffffffu)))
        atomicExch(bad, 1);
    if (__float_as_uint(__uint2float_rd(0xffffffffu)) != __float_as_uint(cts_uint2float_rd(0xffffffffu)))
        atomicExch(bad, 1);
    if (__float2ll_rn(2.5f) != cts_float2ll_rn(2.5f)) atomicExch(bad, 1);
    if (__float2ll_rn(1.0e20f) != cts_float2ll_rn(1.0e20f)) atomicExch(bad, 1);
    if (__float2ll_rn(nanf.f) != cts_float2ll_rn(nanf.f)) atomicExch(bad, 1);
    if (__float2ll_rn(two63) != cts_float2ll_rn(two63)) atomicExch(bad, 1);
    if (__float2ll_rd(-two63) != cts_float2ll_rd(-two63)) atomicExch(bad, 1);
    if (__float2ull_rn(nanf.f) != cts_float2ull_rn(nanf.f)) atomicExch(bad, 1);
    if (__float2ull_rn(-1.0f) != cts_float2ull_rn(-1.0f)) atomicExch(bad, 1);
    if (__float2ull_rn(1.0e20f) != cts_float2ull_rn(1.0e20f)) atomicExch(bad, 1);
    long long wide = (1LL << 53) + 1;
    long long tie = (1LL << 53) + (1LL << 29);
    long long lo = (long long)0x8000000000000000ULL;
    if (__float_as_int(__ll2float_rn(wide)) != __float_as_int(cts_ll2float_rn(wide))) atomicExch(bad, 1);
    if (__float_as_int(__ll2float_ru(wide)) != __float_as_int(cts_ll2float_ru(wide))) atomicExch(bad, 1);
    if (__float_as_int(__ll2float_rd(-tie)) != __float_as_int(cts_ll2float_rd(-tie))) atomicExch(bad, 1);
    if (__float_as_int(__ll2float_rn(lo)) != __float_as_int(cts_ll2float_rn(lo))) atomicExch(bad, 1);
    if (__float_as_int(__ll2float_rn(9223372036854775807LL)) != __float_as_int(cts_ll2float_rn(9223372036854775807LL)))
        atomicExch(bad, 1);
    if (__float_as_uint(__ull2float_rn(0xffffffffffffffffull)) != __float_as_uint(cts_ull2float_rn(0xffffffffffffffffull)))
        atomicExch(bad, 1);
    if (__float_as_uint(__ull2float_rd(0xffffffffffffffffull)) != __float_as_uint(cts_ull2float_rd(0xffffffffffffffffull)))
        atomicExch(bad, 1);
    if (__double2int_rz(nand.d) != cts_double2int_rz(nand.d)) atomicExch(bad, 1);
    if (__double2int_rz(1.9) != cts_double2int_rz(1.9)) atomicExch(bad, 1);
    if (__double2int_rz(1.0e20) != cts_double2int_rz(1.0e20)) atomicExch(bad, 1);
    if (__double2uint_rz(nand.d) != cts_double2uint_rz(nand.d)) atomicExch(bad, 1);
    if (__double2uint_rz(-1.2) != cts_double2uint_rz(-1.2)) atomicExch(bad, 1);
    if (__double2ll_rz(nand.d) != cts_double2ll_rz(nand.d)) atomicExch(bad, 1);
    if (__double2ll_rz(1.0e20) != cts_double2ll_rz(1.0e20)) atomicExch(bad, 1);
    if (__double2ull_rz(nand.d) != cts_double2ull_rz(nand.d)) atomicExch(bad, 1);
    if (__double2ull_rz(1.0e20) != cts_double2ull_rz(1.0e20)) atomicExch(bad, 1);
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
""",
        )

    def test_directed_float_helpers_agree_with_the_intrinsics(self) -> None:
        helpers = "\n".join(
            [
                FARITH_PACK.strip("\n"),
                FARITH_ADD.strip("\n"),
                FARITH_MUL.strip("\n"),
                FARITH_FMA.strip("\n"),
                *(helper.strip("\n") for _name, helper in FARITH_HELPERS.values()),
            ]
        )
        self._nvcc_mismatch(
            "farith",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helpers}
__global__ void compare(int* bad) {{
    unsigned vals[14] = {{
        0, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x21800000u, 0x33800000u,
        0x7f000000u, 0x00000001u, 0x007fffffu, 0x00800000u, 0x3f800001u,
        0x7f800000u, 0xff800000u, 0x7fc00000u
    }};
    for (int i = 0; i < 14; ++i) for (int j = 0; j < 14; ++j) {{
        float x = __uint_as_float(vals[i]);
        float y = __uint_as_float(vals[j]);
        if (__float_as_uint(__fadd_rn(x, y)) != __float_as_uint(cts_fadd_rn(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fadd_rz(x, y)) != __float_as_uint(cts_fadd_rz(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fadd_ru(x, y)) != __float_as_uint(cts_fadd_ru(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fadd_rd(x, y)) != __float_as_uint(cts_fadd_rd(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fsub_rn(x, y)) != __float_as_uint(cts_fsub_rn(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fsub_rz(x, y)) != __float_as_uint(cts_fsub_rz(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fsub_ru(x, y)) != __float_as_uint(cts_fsub_ru(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fsub_rd(x, y)) != __float_as_uint(cts_fsub_rd(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fmul_rn(x, y)) != __float_as_uint(cts_fmul_rn(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fmul_rz(x, y)) != __float_as_uint(cts_fmul_rz(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fmul_ru(x, y)) != __float_as_uint(cts_fmul_ru(x, y))) atomicExch(bad, 1);
        if (__float_as_uint(__fmul_rd(x, y)) != __float_as_uint(cts_fmul_rd(x, y))) atomicExch(bad, 1);
    }}
    unsigned hard[][3] = {{
        {{0x3f800001u, 0x3f800001u, 0xbf800002u}},
        {{0x7f800000u, 0x3f800000u, 0xff800000u}},
        {{0, 0x7f800000u, 0x3f800000u}},
        {{0x7fc00000u, 0x3f800000u, 0x3f800000u}},
        {{0xffc00000u, 0x3f800000u, 0x3f800000u}},
        {{0x3f800000u, 0x21800000u, 0x3f800000u}},
        {{0x3f800000u, 0x33800000u, 0x3f800000u}},
        {{0x00800000u, 0x00800000u, 0}},
        {{0x7f7fffffu, 0x3f800000u, 0x7f7fffffu}},
        {{0x21800000u, 0x00800000u, 0}}
    }};
    for (int n = 0; n < 10; ++n) {{
        float x = __uint_as_float(hard[n][0]);
        float y = __uint_as_float(hard[n][1]);
        float z = __uint_as_float(hard[n][2]);
        if (__float_as_uint(__fmaf_rn(x, y, z)) != __float_as_uint(cts_fmaf_rn(x, y, z))) atomicExch(bad, 1);
        if (__float_as_uint(__fmaf_rz(x, y, z)) != __float_as_uint(cts_fmaf_rz(x, y, z))) atomicExch(bad, 1);
        if (__float_as_uint(__fmaf_ru(x, y, z)) != __float_as_uint(cts_fmaf_ru(x, y, z))) atomicExch(bad, 1);
        if (__float_as_uint(__fmaf_rd(x, y, z)) != __float_as_uint(cts_fmaf_rd(x, y, z))) atomicExch(bad, 1);
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
""",
        )

    def test_saturate_helper_agrees_with_the_intrinsic(self) -> None:
        helper = SATURATE_HELPERS["__saturatef"][1].strip("\n")
        self._nvcc_mismatch(
            "sat",
            f"""\
#include <cstdio>
#include <cuda_runtime.h>
{helper}
__global__ void compare(int* bad) {{
    union {{ float f; unsigned u; }} nanf;
    nanf.u = 0x7fc00000u;
    if (__float_as_uint(__saturatef(-1.0f)) != __float_as_uint(cts_saturatef(-1.0f))) atomicExch(bad, 1);
    if (__float_as_uint(__saturatef(2.0f)) != __float_as_uint(cts_saturatef(2.0f))) atomicExch(bad, 1);
    if (__float_as_uint(__saturatef(0.5f)) != __float_as_uint(cts_saturatef(0.5f))) atomicExch(bad, 1);
    if (__float_as_uint(__saturatef(-0.0f)) != __float_as_uint(cts_saturatef(-0.0f))) atomicExch(bad, 1);
    if (__float_as_uint(__saturatef(nanf.f)) != __float_as_uint(cts_saturatef(nanf.f))) atomicExch(bad, 1);
    if (__float_as_uint(__saturatef(0.0f)) != __float_as_uint(cts_saturatef(0.0f))) atomicExch(bad, 1);
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
""",
        )


if __name__ == "__main__":
    unittest.main()







