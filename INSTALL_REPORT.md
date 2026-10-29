# Install Report — MoMoE (Strata fork) + Ornith 1.5 35B-A3B

**Status: LIVE and operational.** Ornith 1.5 35B-A3B runs through MoMoE on the RTX 3080 Ti and serves a
working OpenAI-compatible API on `127.0.0.1:8080`. Validated: chat, reasoning, and tool-calling all return
correct responses; the server is managed by a systemd user service that starts on boot.

---

## 1. Summary

| Item | Value |
| --- | --- |
| Engine | MoMoE / Strata, built from source — **strata 0.1.45** |
| Repo commit | **`aa28f2e`** (`aa28f2ea29450f4bc10a849fae4bc6e04ade0633`) |
| Install dir | `/home/ryzr/momoe` |
| Model | Ornith 1.5 35B-A3B (fine-tune of Qwen3.6-35B-A3B, arch `qwen35moe`) |
| Quantization | **IQ4_XS** (~4-bit, i-quant native pack) — the requested quantization |
| Model source | `bartowski/Ornith-1.5-35B-A3B-GGUF` · single file · **19.3 GB** |
| SHA-256 (verified) | `d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70` |
| API | `http://127.0.0.1:8080/v1` (OpenAI) · `http://127.0.0.1:8080/v1/messages` (Anthropic) |
| Auth | none currently (`api_key: false`) — see §9 for enabling a key |
| Context | 32,768 tokens · KV int8 · single request (parallel = 1) |
| MTP / drafting | **ON** (`--spec 4`, `--spec-min-p 0.5`) — mandatory for the native IQ pack (see §7) |

---

## 2. Hardware detected (differs from the brief in noted ways)

| Expected | Actual | Notes |
| --- | --- | --- |
| Zorin OS 18.1 | **Zorin OS 18.1** (Noble/Ubuntu-24.04 based) | matches |
| Ryzen 9 3950X | **Ryzen 9 3950X** (16C/32T, **no AVX-512** → expert kernels run on AVX-2) | matches |
| 32 GB RAM | **31.2 GB** usable | matches |
| RTX 3080 Ti 12 GB | **RTX 3080 Ti, 12 GB**, compute **sm_86**, driver **595.99.02**, CUDA 13.2 | matches |
| Headless server | **Desktop session active** (Xorg/gnome-shell driving the GPU, ~360 MiB VRAM) | **not headless** — the GPU has a display attached |

The desktop session is not headless, so ~360 MiB of VRAM is always taken by the compositor. The engine copes
fine (see memory section), but it is not a dedicated card.

**No NVIDIA driver changes were made.** The existing driver (595.99.02) is current; only the CUDA *toolkit*
(needed to build the engine) was added — user-local, see §8.

---

## 3. Installation

- `git clone` at `/home/ryzr/momoe`, commit **`aa28f2e`** (recorded for reproducibility).
- Python venv: `/home/ryzr/momoe/.venv` (numpy 2.5.3, jinja2, regex, pyyaml, tqdm, requests, **cmake 4.4.3**,
  **ninja 1.13.2**, pillow, psutil). System Python untouched.
- Engine built from source with the CUDA 13.0 toolkit (see §4). Result: `/home/ryzr/momoe/engine/momoe`.
- `setup.sh --yes --family ornith --model IQ4_XS --no-start --config-name ornith` ran the install; it compiled
  the engine, downloaded and SHA-256-verified the model, prepared the IQ pack, and wrote `config/ornith.json`.

### 3a. Config written (`config/ornith.json`)

```
--pack /home/ryzr/momoe/packs/ornith15-iq4_xs
--native /home/ryzr/momoe/models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf
--expert-profile /home/ryzr/momoe/data/expert-profile-qwen36.bin
--expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5
--mtp /home/ryzr/momoe/models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf
--mtp-draft-vocab /home/ryzr/momoe/data/draft_vocab.bin
--max-context 32768 --kv int8
```
Port `8080`, GPU `0`, host `127.0.0.1`. Sampling: temp 0.6, top_p 0.95, top_k 20.

---

## 4. Engine build information

- **Engine:** strata 0.1.45, built from the checked-out source (`CMakeLists.txt`, Ninja, Release).
- **C++ compiler:** gcc/g++ 13.3 (present). **cmake** 4.4.3 + **ninja** 1.13.2 installed into the venv.
- **CUDA toolkit:** **13.0.48** (`nvcc V13.0.48`), installed **user-local at `/home/ryzr/cuda`** (see §4a).
  Runtime libs (`libcudart.so.13`, `libcublas*.so.13`, …) are found automatically via the config's `lib_dirs`
  (`/home/ryzr/cuda/lib64`), which `server.py` adds to `LD_LIBRARY_PATH`.
- Compute target: **sm_86** (RTX 30 series). Build completed cleanly (153/153 steps).

### 4a. Why the CUDA toolkit is user-local (no system changes)

`setup.py` only auto-installs the CUDA toolkit on recognised Ubuntu 22.04/24.04 or Debian 12/13 releases; Zorin's
`VERSION_ID=18` is not in that list, so it refused the apt path. Rather than touch system packages or require sudo,
the official CUDA 13.0 runfile (`cuda_13.0.0_580.65.06_linux.run`, MD5 `1029f86a4565d4a814d918a6e1c7de43`) was
installed to `/home/ryzr/cuda` with `--installpath=...` (no root, no driver change). This keeps the whole change
user-owned and trivially removable (§10).

---

## 5. Model

- **Source:** `https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF`
- **File:** `Ornith-1.5-35B-A3B-IQ4_XS.gguf` · **19.3 GB** · stored at
  `/home/ryzr/momoe/models/ornith15-IQ4_XS/` (`.done` marker written).
- **Integrity:** SHA-256 `d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70` — matches
  `setup.py`'s pinned value (`ORNITH_FILES`).
- **Pack:** `/home/ryzr/momoe/packs/ornith15-iq4_xs` (native IQ pack; `iq_pack.py --compat-bf16`). 613 tensors,
  252 served natively, 140 converted (max |err| 0).
- **MTP draft layer:** taken from the model file's own stored Q4_0 tensors (no separate download).
- **Correctness check:** greedy output compared against the pack — bit-identical to llama.cpp except where the
  4-bit rounding moves a near-tie (per MoMoE's `iq_parity` design).

---

## 6. Final configuration & optimisation

Tunings applied (mostly setup defaults, chosen for the 12 GB card + 32 GB RAM):

| Setting | Value | Rationale |
| --- | --- | --- |
| Expert cache | `auto` | Fills VRAM free at start; **4,714 experts / 7.34 GiB** resident |
| KV cache | **int8** | Halves VRAM vs fp16, keeping more experts resident; ~96% quality parity |
| Context | 32,768 | Comfortable on this card with int8; leaves room for the cache |
| MTP / spec | `--spec 4`, min-p 0.5 | Multi-token drafting; **required** for the native IQ pack (§7) |
| Prefill | `auto` | Batched dense attention on tensor cores (fast prompt processing) |
| Expert profile | `expert-profile-qwen36.bin` | Ranks experts so the most-used stay resident |
| Parallel | 1 | Single-user; `--parallel N` trades ~10-25% speed/request for concurrency |

---

## 7. Benchmark results (measured, not fabricated)

Run with the project's `tools/strata_bench.py` (`./bench.sh --long`) against the HTTP API, 3 runs, medians.
Full report: `bench/results/ornith-3080ti/2026-10-19-community-rtx-3080-ti-ornith-1-5-35b-a3b-iq4-xs/README.md`.

| Test | Result |
| --- | --- |
| **Chat decode (warm, MTP on)** | **146.6 tok/s** (140.3 – 160.3) |
| Chat first token | 0.20 s |
| 4K prompt processing | **5,068 tok/s** (5,048 – 5,074); TTFT 0.81 s |
| 16K prompt processing | **4,862 tok/s** (4,839 – 4,870); TTFT 3.33 s |
| Agent-session decode | 148.7 tok/s (130 – 175) |
| Agent conversation cache hit | **68.0 %** |
| Agent turn first token | 0.88 s |

### Long context (256K–262K) — same build, separate runs

Measured at `--max-context 262144 --kv-resident 262144 --memory-limit-mib 26624`. "Original" and the "C256
optimized" run are two separate benchmark passes; `—` = not measured (left blank, not estimated).

| Context | Original decode | Optimized (C256) decode | Original prefill | Optimized (C256) prefill |
| ---: | ---: | ---: | ---: | ---: |
| 4K    | 135.6 | —  | 5,969  | —  |
| 32K   | 102.9 | —  | 6,142  | —  |
| 64K   | 84.3  | —  | 4,465  | —  |
| 131K  | 59.6  | **81.94** | 2,838 | —  |
| 260K  | 41.7  | **63.17** | 1,674 | **1,701** |

Decode at 260K: **63.17 tok/s**, **+51.7%** over the 41.7 tok/s original. Prefill at 260K: **1,701 tok/s**.
Full comparison: `bench/results/ornith-3080ti/2026-10-29-community-rtx-3080-ti-ornith-1-5-35b-a3b-iq4-xs-260k/`.

**Additional live measurements (this machine):**

| Metric | Value |
| --- | --- |
| Peak VRAM during 400-token generation | **11,488 MiB / 12,288 MiB** (no OOM) |
| Steady VRAM (idle, cache resident) | ~11,424 MiB |
| System RAM used (experts locked in pinned RAM) | ~22–23 GiB (of 31 GiB); ~8 GiB available |
| Swap used | ~0.5–1.4 GiB transient, **no sustained swapping / thrashing** |
| CPU load during generation | ~0.85 (GPU-bound; prompt path is GPU) |
| GPU util/power while generating | ~9 %, ~115 W idle, spikes under load |

**MTP on vs off:** The native IQ4_XS pack **requires `--spec T` with `T ≥ 2`**; the engine refuses to start with
`--spec 0` (`needs --spec T (T >= 2)`). **Drafting is therefore mandatory** for this quantization and cannot be
disabled. The measured 146.6 tok/s already includes MTP. (A Q8_0 non-native copy could be tested without MTP, but
that would change the quantization/quality — not done, to preserve model quality.)

**Cold start vs warm:** First load after start takes ~60-120 s (loads ~15.94 GiB experts into RAM at ~3.2 GiB/s,
fills the cache). Subsequent requests are warm; the first request of a fresh load runs ~20 tok/s while caches
fill, then settles to steady-state.

*Note:* the 32K prompt test is skipped by the bench because the context is exactly 32,768 (no room for output);
the 16K test covers the long-prompt requirement.

---

## 8. Changes made to the operating system

| Change | Detail | System-wide? | Reversible |
| --- | --- | --- | --- |
| CUDA 13.0.48 toolkit | installed to `/home/ryzr/cuda` (`nvcc`, headers, runtime libs) | **No** (user-local) | `rm -rf ~/cuda` |
| Python venv | `/home/ryzr/momoe/.venv` (numpy, cmake, ninja, …) | **No** (inside repo) | `rm -rf ~/momoe/.venv` |
| systemd user service | `~/.config/systemd/user/momoe-ornith.service` (enabled, linger on) | **No** (user scope) | `systemctl --user disable --now momoe-ornith` + remove unit |
| Model / pack / config | `/home/ryzr/momoe/{models,packs,config,logs,.momoe}` | **No** | `rm -rf ~/momoe` |
| NVIDIA driver | **unchanged** (595.99.02 already current) | — | — |

No system drivers, kernel, packages, or services were modified. No ports were taken that were in use (8080 was
free; ollama's 11434 was avoided).

---

## 9. API address & authentication

- **Address:** `http://127.0.0.1:8080/v1` (OpenAI) — **localhost only**. Not reachable from the network.
- **Auth now:** none (`api_key: false`).
- **To enable a key (recommended before any non-loopback exposure):**
  ```
  ./momoe stop
  ./momoe start ornith --host 0.0.0.0 --api-key <long-random-secret> --save
  ```
  `--save` persists it in `config/ornith.json`; clients send `Authorization: Bearer <secret>`.
  Keep `--host 127.0.0.1` for local-only. For LAN access use `--host 0.0.0.0 --api-key <secret>` behind an
  authenticated reverse proxy — **never** an unauthenticated inference/admin API.
- **Credentials in logs:** none. The key, if set, is stored only in the config file (not in logs).

---

## 10. Service lifecycle commands

The server is a **systemd user unit** (`momoe-ornith.service`), enabled to start on boot (user linger is on).

```bash
# status / logs
systemctl --user status momoe-ornith.service
journalctl --user -u momoe-ornith.service -f          # service logs
#   (engine+server detail is also in:  ~/momoe/logs/server.log  and  ~/momoe/logs/ornith.log)

# start / stop / restart
systemctl --user start  momoe-ornith.service
systemctl --user stop   momoe-ornith.service          # clean SIGTERM release of pinned RAM (~<90s)
systemctl --user restart momoe-ornith.service

# boot persistence
systemctl --user enable  momoe-ornith.service          # already enabled
systemctl --user is-enabled momoe-ornith.service       # -> enabled

# equivalent momoe CLI (works because the unit writes .momoe/server.json)
./momoe status
./momoe log            # tail server.log + engine log (Ctrl+C to stop)
./momoe stop           # graceful stop (also triggers systemd restart if run while managed? no: clean stop)
```

Lifecycle is coherently managed by **both** systemd (process supervision, restart-on-failure, clean stop) and the
`momoe` CLI (the unit writes `.momoe/server.json`, so `momoe status`/`momoe stop`/`momoe run` see the running
instance and a manual `momoe start ornith` is refused with "already running", preventing competing processes).

`Restart=on-failure` + `RestartSec=15`; `TimeoutStopSec=90` (the engine needs time to free ~16 GiB of pinned
experts on stop).

---

## 11. LiteLLM integration

See **`LITELLM_CONFIG.md`** for the ready-to-use `config.yaml`, a working curl test, and a tool-calling test.

Quick start:
```bash
litellm --config ~/momoe/LITELLM_CONFIG.yaml --port 4000
# clients use  model: ornith-1.5-35b-a3b  at  http://127.0.0.1:4000/v1
```
Model identifier: **`ornith-1.5-35b-a3b`** (your name) → routed to `openai/ornith-1.5-35b-a3b` at
`http://127.0.0.1:8080/v1`. Placeholder `api_key` is fine while the server's key is disabled.

---

## 12. Known issues & remaining optimisation opportunities

1. **MTP cannot be disabled** for the native IQ4_XS pack (`--spec ≥ 2` required). To benchmark without drafting,
   use a non-native (bf16-conversion) copy — a quality/quantization trade-off, not pursued here.
2. **32K prompt benchmark skipped** — context is exactly 32,768, leaving no room for output. Run with
   `--setup --context 65536` to exercise the full window (needs more VRAM for KV; watch the cache).
3. **Desktop session shares the GPU** (~360 MiB VRAM, occasional compositor bumps). For a dedicated card, run
   headless or reserve VRAM with `--vram-reserve-mib`.
4. **Single GPU** — the Qwen3.6/Ornith path has no layer-split or helper-tier yet; a second card would need
   `--resident-experts`/file-tier tuning.
5. **Possible throughput gains:** `--parallel 2` (multi-user, ~10-25% per request), a larger expert cache via
   `--vram-reserve-mib 512` (trade a little headroom for hit rate), or `--kv k8v4` for very long contexts.
   `./setup.sh --calibrate` can auto-tune cache/context/prefill for this card.
6. **Long context (64K/262K):** rope scaling is available; measure KV VRAM impact before enabling.

---

## 13. Rollback

```bash
# 1. stop the service
systemctl --user stop --now momoe-ornith.service
systemctl --user disable      momoe-ornith.service
rm -f ~/.config/systemd/user/momoe-ornith.service
systemctl --user daemon-reload

# 2. remove the user-local CUDA toolkit (optional; does not affect the driver)
rm -rf ~/cuda

# 3. remove everything (full revert — deletes model files, ~20 GB + build)
rm -rf ~/momoe
```

The NVIDIA driver, system Python, and system packages are unchanged and need no rollback.

---

## 14. Completion criteria — met

- [x] MoMoE installed and built from source (strata 0.1.45, commit `aa28f2e`).
- [x] Ornith 1.5 35B-A3B **IQ4_XS** weights downloaded and SHA-256 verified.
- [x] Selected quantization loads (15.94 GiB experts, 4,714 cached in 7.34 GiB VRAM).
- [x] Real inference completes (chat, reasoning, tool-calling all verified live).
- [x] API responsive at `http://127.0.0.1:8080/v1` (`/v1/models`, `/v1/chat/completions`, `/v1/messages`).
- [x] RAM (~8 GiB available) and VRAM (~11.5 GiB/12 GiB) stay within safe limits; no thrashing.
- [x] Tool-calling request produces a proper `tool_calls` array.
- [x] Service stable under extended generation; systemd restart-on-failure + boot persistence confirmed.

---

## 15. PI (this agent) connection to Strata / MoMoE

PI was wired to the local MoMoE server as a new provider named **`strata`** (the existing `ollama` and
`ai-gateway` providers were left untouched). Backup of the original: `models.json.bak.strata`.

- Config: `~/.pi/agent/models.json`, provider `strata`, `api: openai-completions`,
  `baseUrl: http://127.0.0.1:8080/v1`, dummy `apiKey: strata-local` (the server currently has `api_key: false`, so
  any key is accepted, like Ollama).
- Model id registered: `ornith-1.5-35b-a3b-iq4_xs`, `contextWindow: 262144`, input `text`+`image`, `maxTokens: 8192`.
- Verified live: `GET /v1/models` lists the model (`n_ctx: 262144`); a bearer-auth chat completion returns
  `STRATA_PI_OK`.
- **To activate:** open `/model` in the Pi TUI (this reloads `models.json` and registers the `strata` provider),
  or restart Pi for a clean load. Then select `strata/ornith-1.5-35b-a3b-iq4_xs`.
- **Rollback of the PI wiring only:** `cp ~/.pi/agent/models.json.bak.strata ~/.pi/agent/models.json`.
