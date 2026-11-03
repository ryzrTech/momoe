# Verified AI-XS deployment

The accepted deployment is the paired-query INT8 attention build with the earlier dense-prefill workspace and split-decode fixes. Later experimental kernels have not replaced it. The dedicated server installer reproduces this software profile from source. Validation receipts distinguish the rebuilt engine from the historical executable.

## Measured deployment

Hardware: RTX 3080 Ti (12 GiB), Ryzen 9 3950X, approximately 30 GiB usable system RAM. Ubuntu 26.04.1, CUDA 13.1, NVIDIA 595.99.02, sm_86. Native 262,144-token context, Ornith 1.5 35B-A3B IQ4_XS, INT8 KV, thinking enabled and speculative depth 4.

| Run | Fresh input tokens | Reuse | Prefill tokens/s | Decode tokens/s | Interpretation |
|---|---:|---:|---:|---:|---|
| Earlier instrumented split-decode build |259,944|0|890.5|67.1|Control for the accepted paired trial |
| Accepted paired INT8 build |259,944|0|1,031.9|66.6|Accepted software baseline |
| Fresh release install, 2026-11-03 |259,944|0|1,028.4|66.4|Same 388-token output and message/content digests as accepted trial |
| Recent saved-build control, EXP101 |259,944|0|1,017.2|66.5|Shows ordinary operating-state variation |
| Experimental page-four specialization, EXP101 |259,944|0|1,010.1|66.5|No accepted end-to-end improvement; not installed |

The accepted trial reduced prompt time from 291.912 to 251.898 seconds: 15.88% higher prefill throughput. Attention fell from 262.355 to 221.685 seconds; other stages remained near 30 seconds. Generated output differed in that earlier trial, so the decode difference is not an isolated A/B result. The candidate passed randomized 128K marker retrieval. In EXP101, the candidate and control produced the same 388-token output and message/content digests; neither result establishes a speed improvement beyond the saved build.

The earlier split-decode build reached 69.6 decode tokens/s on a longer generation; another shorter generation measured 62.5. These are workload-dependent measurements, not a universal 70 tokens/s guarantee. Historical roughly 1,700 prefill results are not established fresh 260K controls and are not the installer's performance claim.

The full-context numbers above were measured on the host's separately validated GPU-unlock configuration, 300 W limit and existing cooling policy. The application installer does not install that firmware/module, modify boot, set clocks/power/fans or promise these speeds on another machine. 75°C was a temperature target, not a shutdown threshold.

## Frozen provenance

Original application base: 0937f97586f552136cc25e0a02100df501302c76. The saved accepted executable is 81,080,928 bytes, SHA256:

```
b2c0417032bb9413f118204a3c64a133957e18663bf77ae1dcccebdce054aa85
```

| Accepted input | SHA256 |
|---|---|
| Paired INT8 CUDA source |2e5b8250f5d7953415fb5b8313c5d49242865ced4f42bd51c2d3b90fcb13fc96|
| Dense-workspace prefill source |532b76ad7dbe30e224991b98b44a1557cf68aa6c902f3fe4a3288f3e743bcc12|
| Prompt-attention API header |da90f437602c1581a991da183c19c82389a129fc79edb106554c7146394cd6f5|
| Ornith IQ4_XS GGUF, also used for MTP |d6aef57fa948e9bba3ca4959b3c237ed898c605471f48c73a32cedbd24aabe70|

A fresh source build need not have the original executable's SHA: compiler versions, source paths, link layout and build metadata can change binary bytes. The release must preserve the accepted source, build the selected profile, validate dispatch/numerical behavior and record the new executable's digest. A matching filename is not proof.

The saved accepted checkpoint and previous split-decode rollback remain on the operator's server and Mac. The old rollback script deliberately selects the previous split-decode build; do not mistake it for normal recovery of the paired default. Model weights and operator configuration are not committed to Git.

## Validation and remaining work

The release source built cleanly on AI-XS with CUDA 13.1/sm_86 (153 build steps). The rebuilt executable has SHA256 4eb7cdcc8a0a6a155e49e1bcbfb1ab946b4b5efbd63d979a9d56ba45fca04364. Its paired kernel contains the same 5,040 GPU instruction encoding words as the accepted engine; both object and linked-executable comparisons passed. Small, ragged and full-262,144 FP64 gates passed, followed by exact captured-output replay.

The dedicated installer has 37 focused CPU tests. Another 100 existing setup tests passed; two calibration failures were reproduced on unchanged base commit 0937f975 and excluded from that count. Tests exposed legacy cross-checkout migration and systemd path-quoting defects; both were corrected before release.

Release installation validation is recorded in `bench/results/ai-xs/paired-int8-v1/`. The actual one-script installation loaded native context and completed the fresh 259,944-token request at 1,028.4 prefill / 66.4 decode tokens/s, zero reuse, thinking enabled and a 1,024-token allowance. The generated-token count (388) and message/content digests matched the accepted trial. This reproduces the tune within ordinary operating variation, without claiming a new speedup. Randomized 128K audit-marker retrieval passed. A deliberately failed post-start health check restored every saved configuration/engine/unit/environment-file byte and recovered the prior private service. The controller then restored the original b2c accepted default, verified loaded native context and the actual GPU executable, and confirmed the original configuration/registry were unchanged and the test unit removed. The default service is enabled and user linger is active. No new GPU Xid was recorded during validation.

The canonical local GGUF was reused and rehashed; downloading another 19 GB copy, a pristine OS bootstrap, an actual reboot of the new install, system-unit mode and other GPU/toolkit architectures were not tested. Source installation and user-unit persistence/rollback were validated on this host; the installer leaves its hardware policy alone.

See [installation](INSTALL_SERVER.md), [completed research and remaining experiments](docs/ai-xs/optimization-record.md), and [resume checklist](docs/ai-xs/research-next.md).

The original EXP014 installation report is preserved as [historical material](docs/ai-xs/installation-exp014-historical.md); its old live-state and context claims are superseded here.
