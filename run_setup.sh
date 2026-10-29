#!/bin/bash
# Helper to run momoe setup with the user-local CUDA toolkit on PATH.
export PATH="$HOME/cuda/bin:$PATH"
export STRATA_NVCC="$HOME/cuda/bin/nvcc"
export CUDA_HOME="$HOME/cuda"
export LD_LIBRARY_PATH="$HOME/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$HOME/momoe"
exec ./setup.sh "$@"
