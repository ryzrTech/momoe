# RTX 3080 Ti — Ornith 1.5 35B-A3B IQ4_XS · long-context (256K–262K) benchmark

Context dashboard and comparison for the same build measured at **260K context**, paired against the 32K-context
results in [`INSTALL_REPORT.md`](../../INSTALL_REPORT.md).

- **GPU:** NVIDIA GeForce RTX 3080 Ti · **12 GB VRAM**
- **Model:** Ornith-1.5-35B-A3B (IQ4_XS, ~19 GB), `momoe` engine, CUDA
- **KV:** int8 · **expert cache** auto · **prefill** auto · **MTP/spec:** `--spec 4 --spec-min-p 0.5`
- **Context:** `--max-context 262144 --kv-resident 262144 --memory-limit-mib 26624`

## Comparison: original vs "C256" optimized

"Original" and the "C256 optimized" run are two separate benchmark runs. `—` means that measurement was **not taken**
and is intentionally left blank (not estimated).

| Context | Original decode (tok/s) | Optimized (C256) decode (tok/s) | Original prefill (tok/s) | Optimized (C256) prefill (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 4K    | 135.6 | —  | 5,969  | —  |
| 32K   | 102.9 | —  | 6,142  | —  |
| 64K   | 84.3  | —  | 4,465  | —  |
| 131K  | 59.6  | **81.94** | 2,838 | —  |
| 260K  | 41.7  | **63.17** | 1,674 | **1,701** |

## Highlights

- **Decode at 260K: 63.17 tok/s**, up **+51.7%** from the 41.7 tok/s original (and from 59.6 at 131K the optimized
  build keeps more headroom than the original's 59.6 → 41.7 drop).
- **Prefill at 260K: 1,701 tok/s** (vs 1,674 original) — essentially flat while decode improved sharply.

## Missing measurements

The 4K / 32K / 64K optimized-decode and 4K / 32K / 64K / 131K optimized-prefill cells were not measured; they are
left blank rather than estimated. Only the 131K and 260K optimized-decode and the 260K optimized-prefill numbers exist.
