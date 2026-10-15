# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher checks. They do not call vendor SDKs."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_retarget.dispatch import ROOT, execute, plan
from cuda_retarget.gaps import GapError, apply_admitted, validate


class PlanTests(unittest.TestCase):
    def test_hip_starts_with_hipify(self) -> None:
        steps = plan("hip", Path("a.cu"), Path("out"))
        self.assertEqual(steps[0]["argv"][0], "hipify-perl")
        self.assertEqual(steps[2]["argv"][0], "hipcc")

    def test_maca_compiles_with_cucc(self) -> None:
        steps = plan("maca", Path("a.cu"), Path("out"))
        self.assertEqual(steps[-1]["argv"][0], "cucc")
        self.assertNotIn("hipify-perl", json.dumps(steps))


class GapTests(unittest.TestCase):
    def test_shipped_registry_has_no_admitted_gaps(self) -> None:
        registry = json.loads((ROOT / "gaps" / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(validate(registry), [])
        self.assertEqual(registry["gaps"], [])

    def test_uncited_gap_is_rejected(self) -> None:
        errors = validate(
            {
                "gaps": [
                    {
                        "id": "x",
                        "admit": True,
                        "target": "hip",
                        "find": "cudaMalloc",
                        "replace": "hipMalloc",
                    }
                ]
            }
        )
        self.assertTrue(errors)

    def test_one_replacement_after_the_official_text(self) -> None:
        registry = {
            "gaps": [
                {
                    "id": "malloc",
                    "admit": True,
                    "target": "hip",
                    "census_case": "warp_shuffle",
                    "census_status": "compile-fail",
                    "find": "cudaMalloc",
                    "replace": "hipMalloc",
                }
            ]
        }
        self.assertEqual(
            apply_admitted("cudaMalloc(a);\ncudaMalloc(b);\n", registry, "hip"),
            "hipMalloc(a);\ncudaMalloc(b);\n",
        )

    def test_missing_text_is_an_error(self) -> None:
        registry = {
            "gaps": [
                {
                    "id": "malloc",
                    "admit": True,
                    "target": "musa",
                    "census_case": "shared_atomic",
                    "census_status": "compile-fail",
                    "find": "cudaMalloc",
                    "replace": "musaMalloc",
                }
            ]
        }
        with self.assertRaises(GapError):
            apply_admitted("hipMalloc(a);\n", registry, "musa")


class ExecuteTests(unittest.TestCase):
    def test_fake_hipify_is_called_and_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / "bin"
            bindir.mkdir()
            hipify = bindir / "hipify-perl"
            hipify.write_text("#!/bin/sh\nprintf 'cudaMalloc\\n'\n")
            hipcc = bindir / "hipcc"
            hipcc.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/sh
                    while [ "$1" != "-o" ]; do shift; done
                    shift
                    printf 'binary\\n' > "$1"
                    """
                )
            )
            for tool in (hipify, hipcc):
                tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
            source = root / "a.cu"
            source.write_text("int main(){return 0;}\n", encoding="utf-8")
            env = os.environ.copy()
            env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
            # execute() uses shutil.which, which reads the process PATH.
            old = os.environ.get("PATH")
            os.environ["PATH"] = env["PATH"]
            try:
                report = execute("hip", source, root / "out")
            finally:
                if old is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = old
            self.assertEqual(report["steps"][0]["argv"][0], "hipify-perl")
            translated = (root / "out" / "translated.cu").read_text(encoding="utf-8")
            self.assertEqual(translated, "cudaMalloc\n")
            self.assertTrue((root / "out" / "case").is_file())


class TreeTests(unittest.TestCase):
    def test_vendor_tools_are_not_vendored(self) -> None:
        names = {"hipify-perl", "hipify-clang", "musify-text", "cucc"}
        for path in ROOT.rglob("*"):
            if ".git" in path.parts:
                continue
            self.assertNotIn(path.name, names)


if __name__ == "__main__":
    unittest.main()
