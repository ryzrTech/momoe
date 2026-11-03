# AI-XS optimization record

This is the consolidated November 2026 record for Ornith 1.5 35B-A3B IQ4_XS on an RTX 3080 Ti with 12 GiB VRAM and about 30 GiB usable RAM. It supersedes old reports that called an earlier split-decode build the current deployment. The accepted profile is documented in [the deployment report](../../INSTALL_REPORT.md).

## What has been accomplished

1. **Separated cache reuse from fresh-context work.** Earlier requests described as “260K cold” either contained about 174K tokens or reused substantial prefixes. The historical roughly 1,700 prefill figure is not a verified fresh-260K baseline. Current acceptance records actual input, fresh/reused counts, prompt timing, thinking, output allowance, output and message digests, and speculative acceptance.
2. **Located the dominant stage.** Dense prompt attention grows approximately quadratically with context. After the accepted improvements, a complete full-model trace recorded all 320 paired attention calls, totaling 222.788 seconds and matching the engine's 222.787-second stage measurement. Attention consumes about 88% of prefill GPU time; other stages total roughly 30 seconds. SSD bandwidth, expert placement and host launch overhead do not explain that measured attention time.
3. **Preserved the earlier decode gain.** The dense-prefill workspace and G8 split-decode changes reached 69.6 decode tokens/s on one longer generation, versus 62.5 on a shorter generation. The saved split-decode executable remains a rollback artifact. These workload-dependent figures are not a constant throughput guarantee.
4. **Accepted paired-query INT8 attention.** Adjacent dense queries share K/V work while preserving independent causal widths and odd-query tails. Numerical and long-context checks passed. On the matched instrumented fresh-259,944-token workload, prefill rose from 890.5 to 1,031.9 tokens/s, saving about 40 seconds. This is the software profile packaged by the dedicated installer.
5. **Validated a separate GPU-unlock configuration on the test host.** The operator approved physical recovery and a cold boot. A pinned kernel/image was later made persistent. Earlier same-workload measurements rose from about 841/63.8 prefill/decode to 902/67.1. This is hardware-specific evidence, separate from the paired software improvement; the application installer does not perform unlock or boot changes.
6. **Built reliable engineering checks.** Captured real Q/K/V/scales/page tables and expected output; froze source/object/engine digests; proved replay and deployed kernel machine code match; repaired profiler shutdown coverage; collected stall samples with explicit zero overflow/dropped bytes; joined sampled PCs to source; retained full-context FP64, finite/relative error and bitwise captured-output gates. Local Ornith workers do bounded implementation and analysis, with independent review before acceptance.

## Experiment outcomes

“Rejected” applies to the tested implementation, not every possible implementation of that idea. Kernel microbenchmarks, hardware counters and full-model results are distinct evidence.

| Branch / experiments | Finding | Accepted for default? |
|---|---|---|
| Initial system audit, power/temperature, speculative depths 6/8, larger prefill chunks | Small or inconsistent gains; output/speculative acceptance confounded decode comparisons. Hotter settings did not establish a useful prefill gain. | No |
| Isolated CUDA 13.2 comparison with the older baseline | Prefill differences around 0.1–0.3%; no material steady-state gain established. | No |
| EXP018 workspace and split decode | Corrected expensive dense/workspace behavior and improved decode; retained in accepted source lineage. | Yes |
| EXP019–027 dense attention, FlashAttention and paired-query development | FlashAttention candidate failed required numerical accuracy. Paired INT8 passed and later won the full-model trial. | Paired only |
| EXP028–034 post-unlock cache/workspace investigations and paired full-model validation | Some early probes timed the wrong batch API rather than caller-owned workspace. Corrected dispatch/geometry; accepted paired full-model result. New cache-policy candidate was rejected. | Paired build only |
| EXP035 pipeline, prefetch, scheduling variants | No qualified new gain; existing overlap and no-op flags were not counted as new optimizations. | No |
| EXP036–037 bank/layout audit and wider K/V shared stride | Early sampling-overflow and randomized-page-layout confounds corrected. Stride-80 K/V layout was about 5.1% slower at 131K. | No |
| EXP038–040 score exchange vectorization | Roughly 3–4% faster random-input kernels, fewer shared store conflicts and numerical passes. Typed alignment/object provenance corrected. Full model 1,030.0 versus 1,031.9 prefill: no end-to-end gain. | No |
| EXP041–046 dispatch tracing and actual-input capture/replay | Recorded workspace INT8 paired dispatch; no evidence of the proposed fallback in observed calls. Repaired unsafe initial capture code. Actual late-input replay showed about 3.7% kernel gain with bitwise outputs. | Diagnostic evidence |
| EXP047–062 resource, sustained replay, full-model pairs, telemetry and counter work | Sustained score-vector gain persisted near 3.6%; full-model gains remained weak/inconsistent. Corrected worker telemetry windows and overstatements about power causality. | No new default |
| EXP063–068 probability vector exchange | Additional roughly 0.6–1.1% kernel gain over score vectorization. One full-model pair 1,042.6 versus 1,032.1 prefill, same output, but repeatability unproven. | Saved, not selected |
| EXP069–078 timeline tooling and coverage | Early trace lost the final 23 seconds. Corrected collector shutdown/flush produced complete coverage of all 320 calls. | Diagnostic evidence |
| EXP079–084 eight-warp redesign | Initial relative-accuracy failure fixed by integer merging before floating-point conversion. Corrected version passed full FP64/captured bitwise checks but was about 61% slower despite 116 versus 168 registers. | Rejected |
| EXP085–089 exact code, PC stalls and source mapping | Exact deployed/replay encoding equality; minimal PC collection explicitly overflow-free; source mapping and aggregate sums verified. Samples are not elapsed-time fractions or a causal proof. | Diagnostic evidence |
| EXP090–092 independent MMA issue order | Arithmetic and accuracy preserved; final matched pairs showed no useful gain. | Rejected |
| EXP093–097 page-four specialization | Removed repeated runtime page arithmetic, retained exact generic fallback code and all accuracy gates. Three 55°C-start replay pairs saved 3.98%, 4.04%, 4.16% kernel time. | Experimental only |
| EXP098–101 full-model specialization validation | Setup-only hash discrepancy did not reproduce. First pair lacked a baseline after cooldown timeout. Corrected pair: candidate 1,010.1/66.5 versus saved 1,017.2/66.5, fresh 259,944 tokens and identical outputs. | No |
| EXP102–103 prefill-only telemetry review | Corrected averages that included startup/quality tests. Candidate clocks averaged 1,300.7 versus 1,367.2 MHz and were lower in all four elapsed-time quartiles; both drew about 298 W. Equal cool starts did not equal loaded clocks. Causality remains unresolved. | Diagnostic evidence |
| EXP104 same-input, in-request A/B diagnostic | Host-only code reviewed, compiled and linked. Both object and linked-engine GPU instructions equal the frozen generic/specialized kernels. Timer, event creation, timestamp and fallback bugs caught before execution. | Not run; not a deployment |
| EXP105–106 numerical and full-model diagnostic runners | Prepared with restoration/guard controls, but no GPU numerical gates or in-request model trial have run. | Pending |

## What is saved but not selected

- Original EXP014 reference executable, saved split-decode rollback and the accepted paired INT8 deployment/checkpoint.
- Score-vector and combined score/probability-vector candidates, with their qualified kernel gains and inconclusive full-model results.
- Page-four specialization, full numerical evidence, three matched replay pairs and an isolated linked engine; no full-model gain accepted.
- Corrected eight-warp and MMA-order candidates as negative evidence; do not repeat them unchanged.
- EXP104 A/B diagnostic source, object, linked engine and exact instruction proofs; it must never be installed as the default engine.

The source profile in this release uses the accepted paired kernel, not the page-four specialization, probability-vector candidate, diagnostic instrumentation or rejected eight-warp kernel. Private binary checkpoints and raw profiler archives remain with the operator. They contain machine-specific paths/configuration and are not copied wholesale into the public repository.

## Remaining work and limits

The dominant stage is known; its complete source-level root cause and a further repeatable full-model speedup are not established. Counters show several contributors—tensor operations, score exchange, reconstruction, address dependencies and synchronization—rather than proving one fixable dominant instruction.

Next, finish the [same-input A/B gates and experiment](research-next.md). Its purpose is to distinguish actual layer/context behavior from separate-run operating-state differences. Then choose a source change from that evidence. Larger query tiling, K/V reuse and accumulation redesign remain incompletely explored. Lower register count, fewer conflicts or fewer instructions alone do not guarantee higher throughput.

Unverified areas include a broad workload distribution, long production soak, another power cycle after persistent boot-entry selection, alternate kernels/drivers, memory overclock/undervolt/custom fan controls, other GPU architectures, multi-GPU and non-CUDA execution of this dedicated tune. No alternative model weights, weaker numerical tolerances, sparse/windowed approximation or reduced native context were accepted. Many of those are outside this profile's scope.

There is no verified promise of 1,700 fresh-context prefill tokens/s on this hardware. Eliminating all remaining non-attention time would give only an idealized roughly 13.5% throughput increase. Large further gains require changing attention itself or establishing another measured limiting mechanism.
