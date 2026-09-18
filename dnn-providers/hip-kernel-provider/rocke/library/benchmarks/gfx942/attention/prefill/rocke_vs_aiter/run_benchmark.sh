#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIBS="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROOT="$(cd "$ROCM_LIBS/.." && pwd)"

source "$ROOT/env/rocke_env.sh"
cd "$ROCM_LIBS/dnn-providers/hip-kernel-provider"

python "$SCRIPT_DIR/benchmark.py"
