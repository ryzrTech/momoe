#!/usr/bin/env bash
# setup-server.sh - turn a fresh Ubuntu machine (NVIDIA driver already installed) into a dedicated, always-on MoMoE AI
# server running Ornith-1.5-35B-A3B with the best-recorded tuned config. Everything after the driver runs as your user,
# no sudo. See INSTALL_SERVER.md for the write-up.
#
# Override anything with an env var, e.g.:
#   SERVER_HOST=0.0.0.0 SERVER_API_KEY="$(openssl rand -hex 32)" SERVER_USER_SERVICE=0 ./setup-server.sh
set -euo pipefail

: "${SERVER_MODEL:=ornith}"                 # model family
: "${MODEL_SIZE:=IQ4_XS}"                    # quantized size (downloaded later, NOT shipped in this repo)
# The tuned EXP-014 260K config uses IQ4_XS (~19.3 GB). If your GPU cannot fit that at 262144 context, try the
# lighter q4_k_m model first:  MODEL_SIZE=q4_k_m ./setup-server.sh   (it runs; 260K tuning is best on IQ4_XS).
: "${SERVER_PORT:=8080}"
: "${SERVER_HOST:=127.0.0.1}"                # 0.0.0.0 to reach on the LAN
: "${SERVER_API_KEY:=}"                       # required from clients when the host is exposed
: "${SERVER_USER_SERVICE:=1}"                  # 1 = systemd USER service (no sudo); 0 = system service (sudo)
: "${CLONE_URL:=https://github.com/ryzrTech/momoe.git}"

here() { cd "$(dirname "$(readlink -f "$0")")" && pwd; }
log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
err() { printf '\n\033[1;31m%s\033[0m\n' "$*" >&2; }

cd "$(here)"

# ------------------------------------------------------------------ 0. prerequisites
command -v nvidia-smi >/dev/null 2>&1 || {
  err "NVIDIA driver is not present. Install it first (the only sudo step): sudo apt install -y nvidia-driver-580"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { err "nvidia-smi failed - is the driver actually working?"; exit 1; }
echo "NVIDIA driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)";

if ! python3 -c 'import sys,venv,ensurepip; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
  err "Python 3.10+ with venv is needed. Install python3 and python3-venv, then run ./setup-server.sh again."; exit 1; fi

# ------------------------------------------------------------------ 1. clone (if not here already)
if [ ! -f setup.sh ]; then
  log "Cloning MoMoE"
  git clone "$CLONE_URL" .
  cd "$(here)";
else
  echo "Repo already present in $(pwd)."
fi

# ------------------------------------------------------------------ 2. setup + tuned config (install only, do not start)
log "Setting up the model with the best-tuned flags"
# --save persists the tunings into the config so the service reproduces them exactly.
# --no-start because systemd starts it (see step 3). setup.sh answers its own questions with --yes.
./setup.sh --family "$SERVER_MODEL" --model "$MODEL_SIZE" \
  --spec 4 --spec-min-p 0.5 \
  --kv int8 \
  --expert-cache auto --prefill auto \
  --expert-profile ./data/expert-profile-qwen36.bin \
  --max-context 262144 --kv-resident 262144 --memory-limit-mib 26624 \
  --config-name ornith.json \
  --save --no-start --yes

# ------------------------------------------------------------------ 3. systemd unit
# run-server.sh is the ExecStart: it records the process in .momoe/server.json (so the `momoe` CLI can status/stop it)
# and replaces itself with the server so systemd tracks the server process directly.
log "Installing the systemd ${SERVER_USER_SERVICE:+user} service 'momoe-ai-server'"

UNIT_FILE="$(cat <<EOF
[Unit]
Description=MoMoE inference server - Ornith 1.5 35B-A3B (IQ4_XS)
Documentation=file://$(pwd)/INSTALL_SERVER.md
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
# server.py installs a SIGTERM handler, so systemd's default stop is clean and frees the pinned RAM.
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
# EXP-014 tuned defaults (see INSTALL_REPORT.md, bench/results/ornith-3080ti/): chunk C=256 + fast-merge gives 63
# tok/s @260K on the RTX 3080 Ti (+8.9% vs the C=64 default). Read by the engine at runtime; inherited by
# run-server.sh, so the service reproduces the tuned numbers exactly. Override with --chunk-size / the env vars.
Environment=STRATA_ATTN_CHUNK=256
Environment=STRATA_ATTN_MERGE_FAST=1
Environment=SERVER_PORT=${SERVER_PORT}
Environment=SERVER_HOST=${SERVER_HOST}
${SERVER_API_KEY:+Environment=SERVER_API_KEY=${SERVER_API_KEY}}
ExecStart=$(pwd)/run-server.sh
# Restart on crash; keep a gap so a crashing engine does not spin.
Restart=on-failure
RestartSec=15
# Let the engine free its ~16 GiB of pinned experts on stop.
TimeoutStopSec=90
KillSignal=SIGTERM

[Install]
WantedBy=default.target
EOF
)"

if [ "$SERVER_USER_SERVICE" = "1" ]; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  UNIT_PATH="$UNIT_DIR/momoe-ai-server.service"
  mkdir -p "$UNIT_DIR"
  printf '%s' "$UNIT_FILE" > "$UNIT_PATH"
  loginctl enable-linger "$USER" 2>/dev/null || true   # start on boot without a login (no sudo)
  systemctl --user daemon-reload
  systemctl --user enable --now momoe-ai-server.service
  echo "Enabled (user, linger on). Check with:  systemctl --user status momoe-ai-server.service"
else
  UNIT_PATH="/etc/systemd/system/momoe-ai-server.service"
  sudo mkdir -p /etc/systemd/system
  printf '%s' "$UNIT_FILE" | sudo tee "$UNIT_PATH" >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now momoe-ai-server.service
  echo "Enabled (system service). Check with:  systemctl status momoe-ai-server.service"
fi

# ------------------------------------------------------------------ 4. health check
log "Done"
URL="http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
echo "  API : http://${SERVER_HOST}:${SERVER_PORT}   (host ${SERVER_HOST})"
if [ "$SERVER_API_KEY" ]; then echo "  key : set (SERVER_API_KEY)"; fi
echo "  unit: momoe-ai-server.service (user service: $([ "$SERVER_USER_SERVICE" = 1 ] && echo yes || echo no))"
echo "  check: curl $URL   |   journalctl $([ "$SERVER_USER_SERVICE" = 1 ] && echo '--user -u momoe-ai-server.service -f')"
echo
echo "  Waiting for the engine to come up (this can take 30-90s on first start)..."
for i in $(seq 1 60); do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "  Server is up: $URL"; exit 0;
  fi
  sleep 3
done
echo "  Server did not answer on $URL yet - see logs: journalctl $([ "$SERVER_USER_SERVICE" = 1 ] && echo '--user -u momoe-ai-server.service -f')"
