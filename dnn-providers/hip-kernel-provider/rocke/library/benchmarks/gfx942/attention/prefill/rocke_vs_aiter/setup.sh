#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(git rev-parse --show-toplevel)"
ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
AITER="$ROOT/aiter"
ROCKE="$REPO/dnn-providers/hip-kernel-provider/rocke"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

AITER_REF=cdf6ee88a128c2c160b0512fe37cfb67be161b1d
AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/
TRITON_VER=3.8.0+git4cff872c.rocm10.0.0

mkdir -p "$VENV" "$ENV"

# ----------------------------------------------------------------------
# Host dependencies
# ----------------------------------------------------------------------

apt-get update
apt-get install -y \
  git python3.12 python3.12-venv python3.12-dev python3-pip \
  cmake ninja-build build-essential pkg-config libnuma-dev

# ----------------------------------------------------------------------
# AITER checkout
# ----------------------------------------------------------------------

if [ ! -d "$AITER/.git" ]; then
  git clone https://github.com/ROCm/aiter.git "$AITER"
fi

git -C "$AITER" checkout "$AITER_REF"
git -C "$AITER" submodule update --init --recursive

# ----------------------------------------------------------------------
# Common venv setup
# ----------------------------------------------------------------------

make_venv() {
  local v="$1"

  if [ ! -x "$v/bin/python" ] || \
     ! "$v/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$v"
    python3.12 -m venv "$v"
  fi

  "$v/bin/python" -m pip install -U \
    pip setuptools wheel packaging ninja psutil

  "$v/bin/python" -m pip install \
    --index-url "$AMD_INDEX" \
    'rocm[devel,device-gfx942,libraries]==10.0.0' \
    'torch[device-gfx942]==2.13.0+rocm10.0.0'

  "$v/bin/rocm-sdk" init >/dev/null
}

# ----------------------------------------------------------------------
# ROCKE
# ----------------------------------------------------------------------

make_venv "$VENV/rocke"

"$VENV/rocke/bin/python" -m pip install -e "$ROCKE/platform"
"$VENV/rocke/bin/python" -m pip install -e "$ROCKE/library" --no-deps

cat > "$ENV/rocke_env.sh" <<EOF
source "$VENV/rocke/bin/activate"

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

# ----------------------------------------------------------------------
# AITER
# ----------------------------------------------------------------------

make_venv "$VENV/aiter"

"$VENV/aiter/bin/python" -m pip install \
  -r "$AITER/requirements.txt"

"$VENV/aiter/bin/python" -m pip install \
  --index-url "$AMD_INDEX" \
  --force-reinstall \
  --no-deps \
  "triton==$TRITON_VER"

AITER_USE_SYSTEM_TRITON=1 \
PREBUILD_KERNELS=0 \
  "$VENV/aiter/bin/python" -m pip install -e "$AITER"

MHA="$AITER/op_tests/cpp/mha"

cat > "$ENV/aiter_env.sh" <<EOF
source "$VENV/aiter/bin/activate"

rocm-sdk init >/dev/null

export ROCM_HOME="\$(rocm-sdk path --root)"
export ROCM_PATH="\$ROCM_HOME"
export HIP_PATH="\$ROCM_HOME"
export PATH="\$(rocm-sdk path --bin):\$PATH"

export HIP_PLATFORM=amd
export AITER_USE_SYSTEM_TRITON=1
export PREBUILD_KERNELS=0
export AITER_ASM_DIR="$AITER/hsa/"

LIB="\$(python -c 'import importlib.util, pathlib; s=importlib.util.find_spec("_rocm_sdk_core"); print(pathlib.Path(list(s.submodule_search_locations)[0]) / "lib")')"

export LD_LIBRARY_PATH="$MHA:\$LIB:\${LD_LIBRARY_PATH:-}"

unset LIB
EOF

bash -n "$ENV/aiter_env.sh"

# ----------------------------------------------------------------------
# Build AITER FMHA-v3 ASM benchmark
# ----------------------------------------------------------------------

bash -lc "
source '$ENV/aiter_env.sh'
cd '$MHA'

python compile.py --api=fwd_v3

HIPCC=\"\$(rocm-sdk path --bin)/hipcc\"

\"\$HIPCC\" \
  -I'$AITER/3rdparty/composable_kernel/include' \
  -I'$AITER/3rdparty/composable_kernel/example/ck_tile/01_fmha/' \
  -I'$AITER/csrc/include' \
  -std=c++20 \
  -O3 \
  -DUSE_ROCM=1 \
  -DENABLE_CK=1 \
  -DCK_TILE_FMHA_FWD_SPLITKV_API=0 \
  --offload-arch=native \
  -L'$MHA' \
  -lmha_fwd \
  '$MHA/benchmark_mha_fwd.cpp' \
  -Wl,-rpath,'$MHA' \
  -o '$MHA/fwd.exe'
"

echo
echo "Setup complete."
echo "ROCKE env: source $ENV/rocke_env.sh"
echo "AITER env: source $ENV/aiter_env.sh"
echo "Run: ./run_benchmark.sh"

