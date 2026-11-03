# Resume the attention investigation

Research remains paused. Publishing a reproducible accepted installation does not promote the unrun diagnostic or complete the attention root-cause goal.

## Immediate unfinished experiment

The next diagnostic runs original generic page-size handling (PS0) and page-four specialization (PS4) on identical actual attention inputs within one model request. Ten selected eligible call ordinals are 0,9,80,89,160,169,240,249,310,319. Kernel order alternates. Independent CUDA-event pairs exclude output transfers; both complete outputs must match byte for byte. Other eligible calls use PS0. This is an intrusive diagnostic, not a throughput benchmark or production build.

Completed CPU work:

| Artifact | SHA256 |
|---|---|
| Diagnostic source |659238e2752cf610de66b7ab147f8665cc14179d62d71f35f5275fa140b6e75e|
| Object |9cc78a8f010aba57cb21aa21d333c80e4c4e96e0585518e0dc0f8406cad2e402|
| Linked diagnostic engine |6b8454569a6ef9e169ed6db2918dcfd412e34bc82ac8e4c9f5f2330caa3c4377|

Exact instruction comparisons passed for both object and linked engine: generic paired 5,040 encoding words, specialized paired 4,848, non-paired fallback 4,976. Device instructions match the frozen tested versions; no local-memory instructions were found. Source review corrected CPU-copy contamination of timing, missing event creation, incorrect clocks/end timestamps/order mapping, small-request fallback and one-time attribute initialization inside timing.

## Required tests, in order

- [ ] Revalidate canonical artifact, accepted service/configuration, frozen diagnostic digests and absence of conflicting work. Preserve the working deployment and checkpoint.
- [ ] Run small, ragged and full-262,144 FP64 tests, ordinary captured replay and 12-call diagnostic replay. Keep finite outputs, relative `1e-4` and `FP64 error <= max(4 * old error, 1e-6 * reference scale)` unchanged. Captured outputs must remain exact. Check both replay diagnostic records (ordinals 0 and 9), opposite orders, byte counts, event intervals and timestamps.
- [ ] Only after those pass, run the private native-262,144 full-model diagnostic on the established 259,944-fresh-token, zero-reuse, thinking-enabled request with 1,024 output allowance. Validate the expected 388-token output and message/content digests. Require exactly ten correctly ordered records and every bitwise comparison to pass.
- [ ] Independently inspect raw events and approximately 200 ms telemetry windows for each algorithm across layer/context positions. CPU output copies can alter cache/state; alternating order does not eliminate every bias. Do not infer causality or a normalized speedup from clock averages alone.
- [ ] Restore and verify the actual selected executable digest, full-context health, unchanged persistent configuration, absent private override, inactive restoration timer and no GPU errors after each test.
- [ ] Choose the next implementation from the result; then repeat numerical, actual-input and matched fresh full-model/semantic/decode acceptance. Preserve checkpoint and rollback before any promotion.

Operator artifacts are in `exp104-page4-inline-ab`, prepared numerical manifest/runner in `exp105-page4-inline-gates`; the planned model runner is EXP106. These directories are local research artifacts, not prerequisites for installing the released profile. The operator's `Resume Attention Investigation.md` and `task-state.json` contain full paths and runnable supervisor controls. No GPU test from EXP105/106 has been performed as of this checkpoint.

## Opportunities after the diagnostic

1. **Larger dense-attention redesign:** query tiling and K/V reuse, accumulation/warp ownership and synchronization. Preserve precision, causality, ragged rows and resource admission. This is the highest-potential, highest-cost unexplored branch.
2. **Conversion and packing:** reduce repeated query/probability conversion or three-part work only after measuring its contribution. Previously vectorized exchange has small kernel gains but no dependable model win.
3. **Dependencies and overlap:** address demonstrated stalls with a real source change. Do not repeat already-present prefetching, ineffective flags or rejected MMA/warp variants unchanged.
4. **Operating-state mechanism:** quantify sustained clock/power/cooling effects if the A/B evidence supports it; frequency-limit flags are not proof of bottleneck causality. Any hardware-policy experiment needs its own explicit scope and recovery plan.
5. **Other stages:** roughly 12% of current GPU prefill time remains outside attention. Optimize measured contributors, recognizing their smaller whole-request ceiling.

Every investigation needs a time cap, concrete patch or measurement, checkpoint, raw logs and independent review. End a branch when the evidence rejects it; do not turn profiling into an indefinite substitute for engineering.
