# cuda-retarget

One command to run CUDA on AMD, Hygon, Moore Threads, and MetaX. Calls HIPIFY, musify, and cu-bridge, and only fills API gaps those tools miss.

`cuda-retarget` does not reimplement those tools and does not ship their source. It invokes whatever official binary is on `PATH`, then applies a source replacement only when `gaps/registry.json` admits that replacement. An admitted gap must name a [cuda-census](https://github.com/Tes-bo/cuda-census) case and a failure status from that case (`compile-fail`, `run-fail`, `numeric-mismatch`, or `unparsed-stdout`). The shipped registry admits nothing.

This project is not affiliated with NVIDIA, AMD, Hygon, Moore Threads, or MetaX. It does not score itself inside cuda-census. cuda-census runs the official tools directly and refuses a binary named `cuda-retarget`.

## Targets

| `--target` | Official tool | What `run` does |
| --- | --- | --- |
| `hip` | HIPIFY, then `hipcc` | `hipify-perl` writes a translation, admitted gaps may edit it, `hipcc` compiles it. Use the Hygon DTK `hipcc` when that is the `hipcc` on `PATH`. |
| `musa` | musify, then `mcc` | The source is copied, `musify-text --inplace` rewrites the copy, admitted gaps may edit it, `mcc` compiles it. |
| `maca` | cu-bridge | cu-bridge does not emit another dialect. Admitted gaps may edit a copy of the original CUDA file, then `cucc` compiles that copy. With an empty gap list the copy is the original source. |

Command lines live in `targets.json`.

## Run

```bash
python3 -m cuda_retarget self-check
python3 -m cuda_retarget gaps
python3 -m cuda_retarget run --target hip --input kernel.cu --output-dir out --dry-run
python3 -m cuda_retarget run --target hip --input kernel.cu --output-dir out
python3 -m cuda_retarget run --target musa --input kernel.cu --output-dir out
python3 -m cuda_retarget run --target maca --input kernel.cu --output-dir out
```

`--dry-run` prints the steps and does not execute them. A missing official tool, a refused gap, or a non-zero tool exit status exits 2. The compiled binary is `out/case`. This command does not launch that binary.

Host-side tests, with no GPU and no vendor SDK:

```bash
python3 -m unittest discover -s tests -v
```

## Admitting a gap

Leave the registry empty until cuda-census has a JSON result for the official tool on that case, and the status is a failure. Then add one object:

```json
{
  "id": "hip-example",
  "admit": true,
  "target": "hip",
  "census_case": "warp_shuffle",
  "census_status": "compile-fail",
  "find": "the literal text in the official output",
  "replace": "the replacement"
}
```

`find` is replaced once, so the edit stays reviewable. If the official output does not contain `find`, the run stops. A gap with `admit` set and no `census_case` is rejected. Do not admit a gap because a vendor toolchain is missing; that is `toolchain-missing`, and it is not a license to rewrite the source.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
