#!/usr/bin/env bash
# run-server.sh - ExecStart for the systemd unit. Records the running server in .momoe/server.json (same PID, since we
# exec the server next) so the `momoe` CLI can status/stop it, then replaces itself with the server so systemd tracks
# the server process directly. Config/port/host/key come from env (set by the unit) or sane defaults.
#
#   SERVER_CONFIG  -> config file        (default: <repo>/config/ornith.json)
#   SERVER_PORT    -> listening port     (default: 8080)
#   SERVER_HOST    -> bind address       (default: 127.0.0.1)
#   SERVER_API_KEY -> required client key (optional)
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CFG="${SERVER_CONFIG:-$ROOT/config/ornith.json}"
PORT="${SERVER_PORT:-8080}"
HOST="${SERVER_HOST:-127.0.0.1}"
LOG="$ROOT/logs/server.log"
mkdir -p "$ROOT/.momoe" "$ROOT/logs"

# model name, pulled from the config so the status/recorded info is accurate
MODEL_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_name",""))' "$CFG" 2>/dev/null || true)"

cat > "$ROOT/.momoe/server.json" <<EOF
{"pid": $$, "port": ${PORT}, "config": "${CFG}", "host": "${HOST}", "model": "${MODEL_NAME}", "log": "${LOG}", "started": $(date +%s)}
EOF

# server.py auto-adds the config's lib_dirs to LD_LIBRARY_PATH; set it explicitly too for the engine's CUDA libs.
export LD_LIBRARY_PATH="${ROOT}/.momoe/lib_dirs:${LD_LIBRARY_PATH:-}"

exec "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata \
  --config "$CFG" --port "$PORT" "${SERVER_API_KEY:+--api-key "${SERVER_API_KEY}"}" "$@"
