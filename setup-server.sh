#!/usr/bin/env bash
# One-command installation of the verified Ornith IQ4_XS CUDA server profile.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
exec python3 "$ROOT/tools/setup_server.py" "$@"
