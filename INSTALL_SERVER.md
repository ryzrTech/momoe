# Install the verified Ornith server profile

With a working NVIDIA driver, a fresh checkout needs one application-install command:

```bash
git clone https://github.com/ryzrTech/momoe.git
cd momoe
./setup-server.sh
```

The dedicated installer selects the accepted AI-XS paired INT8 software profile. It builds the engine from the checked-out source, prepares Ornith 1.5 35B-A3B **IQ4_XS**, saves the measured settings, installs a persistent systemd service and checks that the model is loaded at its native **262,144-token context**. It does not install a later experimental kernel.

The reference machine has an RTX 3080 Ti with 12 GiB VRAM and about 30 GiB usable system RAM. Other hardware is not covered by its performance measurements. Leave enough disk space for the roughly 19 GB model, build products, Python environment and toolkit. Model weights are downloaded at installation time or reused from an explicitly supplied local artifact; they are not in Git.

## Prerequisites and scope

Use Linux with systemd and a working NVIDIA driver (`nvidia-smi`). Python with venv, compiler tools, CUDA and the project's dependencies must be available or installed through setup's normal dependency path. Package installation and enabling user linger can require administrator authorization. The installer must report a missing prerequisite rather than claim success.

The installer does **not** change firmware, kernel, NVIDIA driver, GRUB, GPU power limits, clocks or fan controls. The reference benchmark host used a separately validated GPU-unlock/pinned-kernel configuration. Reproducing the application tune does not guarantee identical speed on stock or different hardware. See [measured results and provenance](INSTALL_REPORT.md).

The dedicated tune is for the canonical Ornith IQ4_XS artifact. It must not silently select another quantization, reduce context, disable thinking or enable model fallback to fit a machine. Use the ordinary `setup.sh` separately for other models/backends; those are not the validated dedicated profile.

## Persisted tune

| Setting | Value |
|---|---|
| Model / MTP | Same Ornith 1.5 35B-A3B IQ4_XS GGUF |
| Native context |262,144|
| KV precision |INT8|
| Speculative depth / minimum probability |4 / 0.5|
| Expert cache / prefill |`auto` / `auto`|
| VRAM reserve |896 MiB|
| Thinking |Enabled; retain model behavior |
| Split decode |`STRATA_QSA_SPLIT_ATTN=1`|
| Attention chunk / fast merge |`STRATA_ATTN_CHUNK=256`, `STRATA_ATTN_MERGE_FAST=1`|
| Device ordering |`CUDA_DEVICE_ORDER=PCI_BUS_ID`|

The old explicit KV-resident and 26,624 MiB memory-cap defaults are not part of the accepted runtime configuration. Diagnostic stage timing and inline A/B instrumentation are not enabled. The accepted source is an explicit CUDA build profile; an existing engine built without that profile must be rebuilt rather than accepted from cache.

## Service and access

The default is `momoe-ai-server.service`, a systemd user service, bound to **127.0.0.1:8080**. User linger allows startup at boot without an interactive login. Setup verifies it rather than silently ignoring failure. The service uses the checked-out installation directory, not `/home/ryzr` or a research experiment directory.

```bash
systemctl --user status momoe-ai-server.service --no-pager
curl --fail http://127.0.0.1:8080/health
journalctl --user -u momoe-ai-server.service -f
```

Success requires a loaded model and `max_context: 262144`; an HTTP response or model catalog alone is insufficient. Startup failure returns a nonzero status. The installer stages build/configuration before service cutover and retains the prior working selection on a failed update.

OpenAI-compatible API: `http://127.0.0.1:8080/v1`. Anthropic-compatible API: `http://127.0.0.1:8080/v1/messages`.

LAN access requires an API key:

```bash
SERVER_HOST=0.0.0.0 SERVER_API_KEY='your-generated-secret' ./setup-server.sh
```

Do not paste real keys into an issue, benchmark log or shell transcript. Default loopback operation needs no public network exposure. Host/key overrides must be forwarded as properly quoted arguments and status metadata serialized as JSON.

## Optional environment settings

| Variable | Default | Purpose |
|---|---|---|
| SERVER_GGUF_DIR | unset | Reuse the canonical GGUF in this directory; still verify its digest |
| SERVER_SERVICE_NAME | momoe-ai-server | Safe service basename for a separate installation |
| SERVER_PORT / SERVER_HOST | 8080 / 127.0.0.1 | Listening address |
| SERVER_API_KEY | empty | Required for a non-loopback bind; stored in a private environment file |
| SERVER_USER_SERVICE | 1 | User unit; 0 installs a system unit with administrator authorization |
| CMAKE_BUILD_PARALLEL_LEVEL | 2 | Source-build jobs; positive integer |
| SERVER_HEALTH_TIMEOUT | 180 | Startup deadline in seconds |

Preparation uses an installation-private setup registry and explicitly disables nearby-checkout data migration. Other installations keep their model files and configuration. The --help option describes the entry point without starting an installation. The generated configuration is ignored by Git; a [portable example](docs/ai-xs/ornith-config.example.json) is provided for inspection.

## Updates and recovery

For a new checkout, run the script above. For an existing checkout, preserve local configuration and script edits before pulling. Older versions mistakenly tracked the machine-specific config/ornith.json; copy it outside the checkout and stash those local edits before updating, or install into a new clone. Do not discard model files or the working checkpoint to make Git pull succeed. Then rerun the same installer. Ordinary rebuilds retain the installed profile when STRATA_AIXS_PAIRED_INT8 is unset; an explicit value of 0 selects the ordinary build. The dedicated installer always requires 1. Engine cache metadata distinguishes this profile from ordinary builds, and the source fingerprint includes the tuned source files. A running executable is replaced atomically; service restart follows artifact/configuration preparation. Preserve the installer transaction/previous selection until the health check passes.

The operator's original accepted binary and rollback checkpoints remain separately saved. A new build can have a different binary digest because its compiler, source paths and metadata differ. The release records source and build provenance rather than falsely requiring every build to equal a historical binary SHA.

For performance testing, record actual fresh/reused input tokens, thinking, output allowance, exact engine/profile/model, telemetry and raw engine timing. Do not compare cached/warm runs to fresh 260K runs. [The research record](docs/ai-xs/optimization-record.md) explains accepted, rejected and unrun work; [the next-test checklist](docs/ai-xs/research-next.md) is separate from normal installation.
