# Installing MoMoE as a dedicated AI server

This is the step-by-step for turning a **fresh Ubuntu machine into a dedicated, always-on AI server** that runs the
Ornith-1.5-35B-A3B model. It is written so you can repeat it on another machine later. Everything after the driver is
one script: [`setup-server.sh`](../setup-server.sh).

> Validated on an RTX 3080 Ti (12 GB), 32 GB RAM. The tuned config, engine build and benchmark are in
> [`INSTALL_REPORT.md`](../INSTALL_REPORT.md) and
> `bench/results/ornith-3080ti/2026-10-19-community-rtx-3080-ti-ornith-1-5-35b-a3b-iq4-xs/`.

---

## What you get

A single OpenAI/Anthropic-compatible server on `127.0.0.1:8080`, managed by a **systemd user service** that starts on
boot, restarts on crash, and stops cleanly. No system packages, kernel, or root service are modified.

- **Model:** Ornith-1.5-35B-A3B (IQ4_XS, ~19 GB) on the `momoe` engine.
- **API:** `http://127.0.0.1:8080/v1` (OpenAI) · `http://127.0.0.1:8080/v1/messages` (Anthropic) — **localhost only**.
- **Correctness:** chat, reasoning and tool-calling verified. Full report in `INSTALL_REPORT.md`.

---

## The one step you cannot script: the NVIDIA driver

`setup-server.sh` assumes a working NVIDIA driver is already installed (check with `nvidia-smi`). On a fresh box:

```bash
sudo apt update && sudo apt install -y nvidia-driver-580    # driver 580+
# verify
nvidia-smi
```

That is the only root command. Everything else runs as your user, with no sudo.

> **Ubuntu 26.04:** the project's installer auto-supports Ubuntu 22.04/24.04 (and Debian 12/13). 26.04 is newer, so
> if `setup.sh` refuses its Python, the CUDA toolkit path is user-local (see below) and `python3-venv` is what 26.04
> already ships — the script detects a broken `.venv` and rebuilds it.

---

## One script to rule them all

```bash
git clone https://github.com/ryzrTech/momoe.git
cd momoe
./setup-server.sh
```

`setup-server.sh` will, in order:

1. Check the NVIDIA driver and Python are present (aborts cleanly if not).
2. Run the project's `setup.sh` for the Ornith IQ4_XS model **with your best-tuned flags saved** (`--save`). This is a
   non-interactive `--yes` run — it downloads the model (~19 GB), the MTP draft layer, and builds/uses the ready-made
   CUDA engine for the RTX 30/40/50 series.
3. Install a systemd user unit (`momoe-ai-server.service`), enable **user linger** (so it starts on boot without a
   login), and start it.
4. Print a health-check line.

### Tuning the script (all optional, set as env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SERVER_MODEL` | `ornith` | model family |
| `MODEL_SIZE` | `IQ4_XS` | quantized size |
| `SERVER_PORT` | `8080` | API port |
| `SERVER_HOST` | `127.0.0.1` | `0.0.0.0` to reach on the LAN |
| `SERVER_API_KEY` | *(empty)* | required key from clients (use with `SERVER_HOST=0.0.0.0`) |
| `SERVER_USER_SERVICE` | `1` | `1` = systemd **user** service (no sudo); `0` = **system** service (uses sudo) |
| `CLONE_URL` | the public repo | clone a different copy if you already have one |

Example — LAN access with a key and a system (root) service:

```bash
sudo apt install -y nvidia-driver-580   # the only root step
clone_only=0 SERVER_HOST=0.0.0.0 SERVER_API_KEY="$(openssl rand -hex 32)" \
  SERVER_USER_SERVICE=0 ./setup-server.sh
```

---

## The tuned flags (what "best benchmark" means)

These are saved into the config by `setup-server.sh` (mirroring the config that produced the benchmark in
`INSTALL_REPORT.md`). They are the recorded best run on the 12 GB card:

```
--spec 4 --spec-min-p 0.5                 # multi-token draft layer (required for the native IQ pack)
--kv int8                                 # halves KV VRAM, keeps more experts resident
--expert-cache auto --prefill auto        # auto-fill VRAM, batched dense prompt attention
--expert-profile data/expert-profile-qwen36.bin
--max-context 262144 --kv-resident 262144 --memory-limit-mib 26624
```

Sampling (in the saved config): temperature 0.6, top_p 0.95, top_k 20.

---

## Measured performance (same card, same config)

Run with `./bench.sh --long`, 3 runs, medians — see
`bench/results/ornith-3080ti/2026-10-19-community-rtx-3080-ti-ornith-1-5-35b-a3b-iq4-xs/`:

| Test | Result |
| --- | --- |
| Chat decode (warm, MTP on) | **146.6 tok/s** (140.3 – 160.3) |
| Chat first token | 0.20 s |
| 4K prompt processing | **5,068 tok/s** (5,048 – 5,074); TTFT 0.81 s |
| 16K prompt processing | **4,862 tok/s** (4,839 – 4,870); TTFT 3.33 s |

---

## Service lifecycle

```bash
systemctl --user status     momoe-ai-server.service
journalctl --user -u momoe-ai-server.service -f   # logs
systemctl --user restart    momoe-ai-server.service
systemctl --user stop       momoe-ai-server.service   # clean SIGTERM; frees ~16 GiB pinned RAM (~90s)

./momoe status              # same lifecycle via the CLI (unit writes .momoe/server.json)
```

If you chose a **system service** (`SERVER_USER_SERVICE=0`), use `systemctl status/start/stop momoe-ai-server.service`
(without `--user`) and `sudo` if needed.

---

## Removing it (leaves nothing behind)

```bash
systemctl --user disable --now momoe-ai-server.service
# user service:
loginctl disable-linger "$USER"     # so it won't boot-start again
rm -f ~/.config/systemd/user/momoe-ai-server.service
# system service:
sudo systemctl disable --now momoe-ai-server.service
sudo rm -f /etc/systemd/system/momoe-ai-server.service
sudo systemctl daemon-reload
# then delete the model files: rm -rf models packs
```

## Security notes

- The server listens on `127.0.0.1` by default and is **not** reachable from the network.
- Do **not** expose it beyond localhost without `SERVER_HOST=0.0.0.0` **and** a strong `SERVER_API_KEY`. Even then, put it
  behind a firewall / VPN — this is a single-user model with no account system.
