#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(git rev-parse --show-toplevel)"
ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
ROCKE="$REPO/dnn-providers/hip-kernel-provider/rocke"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/

mkdir -p "$VENV" "$ENV"

# ----------------------------------------------------------------------
# Host dependencies
# ----------------------------------------------------------------------

apt-get update
apt-get install -y \
  git python3.12 python3.12-venv python3.12-dev python3-pip \
  cmake ninja-build build-essential pkg-config libnuma-dev

# ----------------------------------------------------------------------
# ROCKE virtual environment
# ----------------------------------------------------------------------

ROCKE_VENV="$VENV/rocke"

if [ ! -x "$ROCKE_VENV/bin/python" ] || \
   ! "$ROCKE_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$ROCKE_VENV"
  python3.12 -m venv "$ROCKE_VENV"
fi

"$ROCKE_VENV/bin/python" -m pip install -U \
  pip setuptools wheel packaging ninja psutil

"$ROCKE_VENV/bin/python" -m pip install \
  --index-url "$AMD_INDEX" \
  'rocm[devel,device-gfx942,libraries]==10.0.0' \
  'torch[device-gfx942]==2.13.0+rocm10.0.0'

"$ROCKE_VENV/bin/rocm-sdk" init >/dev/null

# ----------------------------------------------------------------------
# Install ROCKE
# ----------------------------------------------------------------------

"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/platform"
"$ROCKE_VENV/bin/python" -m pip install -e "$ROCKE/library" --no-deps

# ----------------------------------------------------------------------
# ROCKE environment activation script
# ----------------------------------------------------------------------

cat > "$ENV/rocke_env.sh" <<EOF
source "$ROCKE_VENV/bin/activate"

rocm-sdk init >/dev/null

export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"

export HIP_PLATFORM=amd
export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23

LIB="\$(python -c 'import importlib.util, pathlib; s=importlib.util.find_spec("_rocm_sdk_core"); print(pathlib.Path(list(s.submodule_search_locations)[0]) / "lib")')"

export ROCKE_COMGR_LIB="\$LIB/libamd_comgr.so.3"
export ROCKE_HIP_LIB="\$LIB/libamdhip64.so.7"
export LD_LIBRARY_PATH="\$LIB:\${LD_LIBRARY_PATH:-}"

unset LIB
EOF

bash -n "$ENV/rocke_env.sh"

echo
echo "ROCKE setup complete."
echo "ROCKE env: source $ENV/rocke_env.sh"