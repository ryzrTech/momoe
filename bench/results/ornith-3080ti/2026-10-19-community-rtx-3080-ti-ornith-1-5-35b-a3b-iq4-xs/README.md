# Community benchmark on NVIDIA GeForce RTX 3080 Ti

Measured on 2026-10-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 3080 Ti, 12 GiB VRAM
- CPU: AMD Ryzen 9 3950X 16-Core Processor (16 cores, 32 threads); installed RAM: 31.2 GiB; storage: not measured
- OS: Linux 7.0.0-38-generic; GPU driver: 595.99.02; CUDA: 13.2
- Momoe: project 0.1.45, commit aa28f2ea; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: ornith-1.5-35b-a3b-iq4_xs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe --pack ornith15-iq4_xs --native Ornith-1.5-35B-A3B-IQ4_XS.gguf --expert-profile expert-profile-qwen36.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp Ornith-1.5-35B-A3B-IQ4_XS.gguf --mtp-draft-vocab draft_vocab.bin --max-context 32768 --kv int8
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Momoe server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 146.6 (140.3-160.3) | 0.20 (0.19-0.23) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 5068.7 (5048.4-5073.9) | not measured | 0.81 (0.80-0.81) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4862.2 (4839.2-4870.6) | not measured | 3.33 (3.32-3.34) |
| agent session, follow-up turns | 14450 (14430-14455) at the last turn | see below | up to 64 | 3 | see below | 148.7 (130.1-174.5) | 0.88 (0.57-0.98) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.0 % (68.0-68.0); earlier-turn tokens read again per session: 276 (256-281) (earlier turns that the cache did not hold; the 13600 (13600-13600) new tokens of the files are read once); first turn (nothing cached) first token: 0.53 s (0.52-0.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1233.5 | 0.518 | 160.1 |
| 1 | 3638 | 602 | 3036 | 4443.8 | 0.769 | 135.7 |
| 2 | 5162 | 3631 | 1531 | 2721.8 | 0.608 | 155.4 |
| 3 | 8575 | 5155 | 3420 | 3880.6 | 0.929 | 160.6 |
| 4 | 11481 | 8568 | 2913 | 3335.6 | 0.918 | 149.1 |
| 5 | 14450 | 11474 | 2976 | 3110.0 | 0.982 | 130.1 |

### Skipped and failed

- skipped 32K: the context is 32768 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 43 s.
