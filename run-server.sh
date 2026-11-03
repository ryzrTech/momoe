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

# security: a bind beyond this PC (0.0.0.0 / a LAN address) is refused without an API key, so a misconfigured server is
# never left reachable on the network unauthenticated. The unit binds 127.0.0.1 by default.
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]] && [ -z "${SERVER_API_KEY:-}" ]; then
  echo "refusing to bind $HOST without a SERVER_API_KEY (it would expose the server on the network)" >&2
  exit 1
fi

# model name, pulled from the config so the status/recorded info is accurate
MODEL_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model_name",""))' "$CFG" 2>/dev/null || true)"

# record the running server atomically with proper JSON escaping (paths are json.dump'ed, not interpolated)
python3 - "$ROOT/.momoe/server.json" "$$" "$PORT" "$CFG" "$HOST" "$MODEL_NAME" "$LOG" <<'JSONPY'
import json, os, sys, time
path, pid, port, cfg, host, model, log = sys.argv[1:8]
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
json.dump({"pid": int(pid), "port": int(port), "config": cfg, "host": host,
           "model": model, "log": log, "started": int(time.time())},
          open(tmp, "w"), indent=1)
os.replace(tmp, path)
JSONPY

# server.py auto-adds the config's lib_dirs to LD_LIBRARY_PATH; set it explicitly too for the engine's CUDA libs.
export LD_LIBRARY_PATH="${ROOT}/.momoe/lib_dirs:${LD_LIBRARY_PATH:-}"

# Build the argument list with a proper argv array: --host is passed explicitly (run-server.sh used to ignore it), and
# --api-key is added as a single argument only when set, never interpolated into a quoted word and never echoed.
args=(--engine strata --config "$CFG" --host "$HOST" --port "$PORT")
if [ -n "${SERVER_API_KEY:-}" ]; then
  args+=(--api-key "$SERVER_API_KEY")
fi
exec "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" "${args[@]}" "$@"
