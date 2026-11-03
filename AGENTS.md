# AGENTS.md

MoMoE (a fork of **[Strata](https://github.com/Niko1221/Strata)**) runs large mixture-of-experts models -- notably
Ornith-1.5-35B-A3B -- on a normal PC: one NVIDIA or AMD graphics card plus system RAM, on Linux or Apple Silicon. It
has a C++/CUDA/HIP engine (`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web
app (`serve/`), and a one-click installer (`setup.py`, started by `setup.sh`).

## Running it as a dedicated AI server

Use `./setup-server.sh` after cloning on Linux/systemd with a working NVIDIA driver. It selects the accepted AI-XS paired INT8 source profile, canonical Ornith IQ4_XS artifact, native 262144 context, INT8 KV, spec 4/min-p 0.5, automatic expert cache/prefill and 896 MiB VRAM reserve. Persistent service flags include split decode, chunk 256 and fast merge. Thinking remains enabled. Unsupported model/backend overrides must fail clearly rather than silently change quantization or context.

See [INSTALL_SERVER.md](INSTALL_SERVER.md) and [INSTALL_REPORT.md](INSTALL_REPORT.md). The installer must build the profile from source, include profile identity in build cache validity, verify loaded/native-context health and roll back failed service cutovers. Default loopback binding; non-loopback requires a key. No kernel/firmware/driver/power/clock changes are part of this application installer.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.

## Contributing a change or report

- Search the open issues and pull requests first, and add to a thread that already covers your point.
- Open an issue with the form that fits (bug report, feature request or question).
- One change per pull request. Say what it changes and what it leaves alone.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).

## Accepted tune and unfinished research

The accepted historical paired INT8 result is 1031.9 prefill / 66.6 decode at 259944 fresh tokens, zero reuse, thinking enabled. The latest control measured 1017.2/66.5; later candidates have no accepted repeatable full-model gain. Do not advertise 1700 fresh prefill or universal 70 decode. Preserve exact canonical weights, native context and numerical gates.

[docs/ai-xs/optimization-record.md](docs/ai-xs/optimization-record.md) records accepted/rejected/saved candidates. [docs/ai-xs/research-next.md](docs/ai-xs/research-next.md) records unrun in-request A/B numerical/full-model tests. These diagnostic kernels must not become an install default. Complete research requires independent code/raw-data review and matched fresh full-model correctness, semantic and decode acceptance; a microbenchmark win is insufficient.
