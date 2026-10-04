#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench

AITER_VENV="$ROOT/venvs/aiter"
ROCKE_VENV="$ROOT/venvs/rocke"
ROCPROF_VENV="$ROOT/venvs/rocprof"

ROCM_VERSION=10.0.0
ROCM_INDEX="https://stable.repo.amd.com/rocm/whl-next/"

echo "============================================================"
echo "1. Check existing workload environments"
echo "============================================================"

[[ -x "$AITER_VENV/bin/python" ]] || {
    echo "ERROR: AITER venv missing: $AITER_VENV"
    exit 1
}

[[ -x "$ROCKE_VENV/bin/python" ]] || {
    echo "ERROR: ROCKE venv missing: $ROCKE_VENV"
    exit 1
}

[[ -x "$AITER_VENV/bin/rocprofv3" ]] || {
    echo "ERROR: AITER rocprofv3 missing"
    exit 1
}

[[ -x "$ROCKE_VENV/bin/rocprofv3" ]] || {
    echo "ERROR: ROCKE rocprofv3 missing"
    exit 1
}

echo
echo "--- AITER rocprofv3 ---"
"$AITER_VENV/bin/rocprofv3" --version

echo
echo "--- ROCKE rocprofv3 ---"
"$ROCKE_VENV/bin/rocprofv3" --version


echo
echo "============================================================"
echo "2. Recreate dedicated rocprof environment"
echo "============================================================"

rm -rf "$ROCPROF_VENV"

python3 -m venv "$ROCPROF_VENV"

PY="$ROCPROF_VENV/bin/python"
PIP="$PY -m pip"

$PIP install --upgrade pip


echo
echo "============================================================"
echo "3. Install ROCm 10 profiler stack"
echo "============================================================"

$PIP install \
    --index-url "$ROCM_INDEX" \
    "rocm[profiler]==$ROCM_VERSION"


echo
echo "============================================================"
echo "4. Install rocprof-compute Python dependencies"
echo "============================================================"

# Locate the requirements.txt installed by the ROCm profiler wheel.
ROCPROF_REQUIREMENTS="$(
    "$PY" - <<'PY'
import pathlib
import _rocm_profiler

root = pathlib.Path(_rocm_profiler.__file__).resolve().parent
req = root / "libexec" / "rocprofiler-compute" / "requirements.txt"

if not req.is_file():
    raise SystemExit(f"requirements.txt not found: {req}")

print(req)
PY
)"

echo "rocprof-compute requirements:"
echo "  $ROCPROF_REQUIREMENTS"

$PIP install -r "$ROCPROF_REQUIREMENTS"


echo
echo "============================================================"
echo "5. Verify profiler environment"
echo "============================================================"

[[ -x "$ROCPROF_VENV/bin/rocprofv3" ]] || {
    echo "ERROR: rocprofv3 wasn't installed"
    exit 1
}

[[ -x "$ROCPROF_VENV/bin/rocprof-compute" ]] || {
    echo "ERROR: rocprof-compute wasn't installed"
    exit 1
}

echo
echo "--- profiler rocprofv3 ---"
"$ROCPROF_VENV/bin/rocprofv3" --version

echo
echo "--- rocprof-compute ---"
"$ROCPROF_VENV/bin/rocprof-compute" --version

echo
echo "--- Python dependencies ---"
$PIP check

echo
echo "--- Important package versions ---"
"$PY" - <<'PY'
import yaml
import numpy
import pandas
import sqlalchemy
import tabulate

print("PyYAML    ", yaml.__version__)
print("NumPy     ", numpy.__version__)
print("Pandas    ", pandas.__version__)
print("SQLAlchemy", sqlalchemy.__version__)
print("Tabulate  ", tabulate.__version__)

try:
    import dash
    print("Dash      ", dash.__version__)
except Exception as e:
    print("Dash check failed:", e)

try:
    import textual
    print("Textual   ", textual.__version__)
except Exception as e:
    print("Textual check failed:", e)
PY


echo
echo "============================================================"
echo "6. Final paths"
echo "============================================================"

echo "rocprof-compute frontend:"
echo "  $ROCPROF_VENV/bin/rocprof-compute"

echo
echo "AITER counter backend:"
echo "  $AITER_VENV/bin/rocprofv3"

echo
echo "ROCKE counter backend:"
echo "  $ROCKE_VENV/bin/rocprofv3"

echo
echo "Expected configuration in profile script:"
cat <<EOF

export ROCM_VER=$ROCM_VERSION

ROCPROF_COMPUTE=$ROCPROF_VENV/bin/rocprof-compute

AITER_ROCPROF=$AITER_VENV/bin/rocprofv3
ROCKE_ROCPROF=$ROCKE_VENV/bin/rocprofv3

EOF

echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"