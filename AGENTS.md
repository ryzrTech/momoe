# AGENTS.md

MoMoE (a fork of **[Strata](https://github.com/Niko1221/Strata)**) runs large mixture-of-experts models -- notably
Ornith-1.5-35B-A3B -- on a normal PC: one NVIDIA or AMD graphics card plus system RAM, on Linux or Apple Silicon. It
has a C++/CUDA/HIP engine (`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web
app (`serve/`), and a one-click installer (`setup.py`, started by `setup.sh`).

## Running it as a dedicated AI server

To turn a fresh Ubuntu machine into a dedicated, always-on server with the best-recorded tuned config, run
**[`setup-server.sh`](setup-server.sh)** (see **[INSTALL_SERVER.md](INSTALL_SERVER.md)**):

1. Install the NVIDIA driver 580+ (the only sudo step): `sudo apt install -y nvidia-driver-580`.
2. `git clone https://github.com/ryzrTech/momoe.git && cd momoe && ./setup-server.sh`.

It installs the Ornith IQ4_XS model with the tuned flags (`--spec 4 --spec-min-p 0.5 --kv int8 --expert-cache auto
--prefill auto --max-context 262144 --kv-resident 262144 --memory-limit-mib 26624`), then installs a systemd user
service (`momoe-ai-server.service`, enabled with linger) and starts it on `127.0.0.1:8080`. Everything after the driver
runs as your user, no sudo. Override access with env vars (`SERVER_HOST`, `SERVER_API_KEY`, `SERVER_USER_SERVICE`, ...);
see `INSTALL_SERVER.md`. The service and its flags are the same ones that produced the benchmark in
`INSTALL_REPORT.md`.

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
