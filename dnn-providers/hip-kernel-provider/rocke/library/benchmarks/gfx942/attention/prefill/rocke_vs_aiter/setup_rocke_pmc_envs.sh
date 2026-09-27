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
echo "--- AITER Python ---"
"$AITER_VENV/bin/python" --version

echo
echo "--- ROCKE Python ---"
"$ROCKE_VENV/bin/python" --version

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

"$PY" -m pip install --upgrade pip


echo
echo "============================================================"
echo "3. Install ROCm 10 profiler stack"
echo "============================================================"

"$PY" -m pip install \
    --index-url "$ROCM_INDEX" \
    "rocm[profiler]==$ROCM_VERSION"


echo
echo "============================================================"
echo "4. Install rocprof-compute Python dependencies"
echo "============================================================"

"$PY" -m pip install \
    pyyaml


echo
echo "============================================================"
echo "5. Verify profiler environment"
echo "============================================================"

ROCPROF="$ROCPROF_VENV/bin/rocprofv3"
ROCPROF_COMPUTE="$ROCPROF_VENV/bin/rocprof-compute"

[[ -x "$ROCPROF" ]] || {
    echo "ERROR: rocprofv3 wasn't installed"
    exit 1
}

[[ -x "$ROCPROF_COMPUTE" ]] || {
    echo "ERROR: rocprof-compute wasn't installed"
    exit 1
}

echo
echo "--- profiler rocprofv3 ---"
"$ROCPROF" --version

echo
echo "--- rocprof-compute ---"
"$ROCPROF_COMPUTE" --version

echo
echo "--- PyYAML ---"
"$PY" -c 'import yaml; print("PyYAML", yaml.__version__)'


echo
echo "============================================================"
echo "6. Find rocprofv3-avail"
echo "============================================================"

if [[ -x "$ROCPROF_VENV/bin/rocprofv3-avail" ]]; then
    ROCPROF_AVAIL="$ROCPROF_VENV/bin/rocprofv3-avail"

elif [[ -x "/opt/rocm/bin/rocprofv3-avail" ]]; then
    ROCPROF_AVAIL="/opt/rocm/bin/rocprofv3-avail"

elif command -v rocprofv3-avail >/dev/null 2>&1; then
    ROCPROF_AVAIL="$(command -v rocprofv3-avail)"

else
    echo "ERROR: rocprofv3-avail not found"
    exit 1
fi

echo "rocprofv3-avail:"
echo "  $ROCPROF_AVAIL"


echo
echo "============================================================"
echo "7. Check GPU architecture"
echo "============================================================"

if command -v rocminfo >/dev/null 2>&1; then

    echo
    echo "--- GPU architecture(s) ---"

    rocminfo 2>/dev/null \
        | grep -E 'Name:[[:space:]]+gfx[0-9]+' \
        | sort -u \
        || true

else
    echo "WARNING: rocminfo not found in PATH"
fi


echo
echo "============================================================"
echo "8. Check PC sampling support"
echo "============================================================"

echo
echo "--- Agents supporting PC sampling ---"
"$ROCPROF_AVAIL" list --pc-sampling

echo
echo "--- Detailed PC sampling configurations ---"

PC_INFO="$("$ROCPROF_AVAIL" info --pc-sampling)"

printf '%s\n' "$PC_INFO"


echo
echo "============================================================"
echo "9. Validate stochastic PC sampling"
echo "============================================================"

if printf '%s\n' "$PC_INFO" | grep -qi "stochastic"; then
    echo "PASS: stochastic PC sampling supported"
else
    echo "ERROR: stochastic PC sampling not reported"
    exit 1
fi

if printf '%s\n' "$PC_INFO" | grep -Eqi 'cycle|cycles'; then
    echo "PASS: cycle-based PC sampling supported"
else
    echo "ERROR: cycle-based stochastic sampling not reported"
    exit 1
fi

if printf '%s\n' "$PC_INFO" | grep -qi "gfx942"; then
    echo "PASS: gfx942 PC sampling configuration found"
else
    echo "WARNING: gfx942 string not present in PC-sampling output"
fi


echo
echo "============================================================"
echo "10. Verify rocprofv3 PC sampling options"
echo "============================================================"

ROCPROF_HELP="$("$ROCPROF" --help 2>&1 || true)"

for OPT in \
    "--pc-sampling-beta-enabled" \
    "--pc-sampling-method" \
    "--pc-sampling-unit" \
    "--pc-sampling-interval"
do

    if printf '%s\n' "$ROCPROF_HELP" | grep -q -- "$OPT"; then
        echo "PASS: $OPT"
    else
        echo "ERROR: rocprofv3 does not advertise $OPT"
        exit 1
    fi

done


echo
echo "============================================================"
echo "11. Verify rocprof-compute PC sampling options"
echo "============================================================"

COMPUTE_HELP="$("$ROCPROF_COMPUTE" profile --help 2>&1 || true)"

for OPT in \
    "--pc-sampling" \
    "--pc-sampling-method" \
    "--pc-sampling-interval"
do

    if printf '%s\n' "$COMPUTE_HELP" | grep -q -- "$OPT"; then
        echo "PASS: rocprof-compute supports $OPT"
    else
        echo "WARNING: rocprof-compute does not advertise $OPT"
    fi

done


echo
echo "============================================================"
echo "12. Final paths"
echo "============================================================"

echo
echo "Dedicated rocprofv3:"
echo "  $ROCPROF"

echo
echo "rocprofv3-avail:"
echo "  $ROCPROF_AVAIL"

echo
echo "rocprof-compute frontend:"
echo "  $ROCPROF_COMPUTE"

echo
echo "AITER Python:"
echo "  $AITER_VENV/bin/python"

echo
echo "ROCKE Python:"
echo "  $ROCKE_VENV/bin/python"

echo
echo "Original AITER rocprofv3:"
echo "  $AITER_VENV/bin/rocprofv3"

echo
echo "Original ROCKE rocprofv3:"
echo "  $ROCKE_VENV/bin/rocprofv3"


echo
echo "============================================================"
echo "13. Recommended configuration for profiling scripts"
echo "============================================================"

cat <<EOF

ROOT=$ROOT

ROCPROF=$ROCPROF
ROCPROF_AVAIL=$ROCPROF_AVAIL
ROCPROF_COMPUTE=$ROCPROF_COMPUTE

AITER_PYTHON=$AITER_VENV/bin/python
ROCKE_PYTHON=$ROCKE_VENV/bin/python

# Existing workload-local backends, if needed:
AITER_ROCPROF=$AITER_VENV/bin/rocprofv3
ROCKE_ROCPROF=$ROCKE_VENV/bin/rocprofv3

EOF


echo
echo "============================================================"
echo "14. Raw stochastic PC sampling example"
echo "============================================================"

cat <<EOF

# ----------------------------------------------------------
# AITER
# ----------------------------------------------------------

mkdir -p "$ROOT/profiles/pc_sampling/aiter"

"$ROCPROF" \\
    --pc-sampling-beta-enabled \\
    --pc-sampling-method stochastic \\
    --pc-sampling-unit cycles \\
    --pc-sampling-interval 65536 \\
    --output-format csv json \\
    --output-directory "$ROOT/profiles/pc_sampling/aiter" \\
    -- "$AITER_VENV/bin/python" YOUR_AITER_SCRIPT.py


# ----------------------------------------------------------
# ROCKE
# ----------------------------------------------------------

mkdir -p "$ROOT/profiles/pc_sampling/rocke"

"$ROCPROF" \\
    --pc-sampling-beta-enabled \\
    --pc-sampling-method stochastic \\
    --pc-sampling-unit cycles \\
    --pc-sampling-interval 65536 \\
    --output-format csv json \\
    --output-directory "$ROOT/profiles/pc_sampling/rocke" \\
    -- "$ROCKE_VENV/bin/python" YOUR_ROCKE_SCRIPT.py

EOF


echo
echo "============================================================"
echo "15. rocprof-compute PC sampling example"
echo "============================================================"

cat <<EOF

# ----------------------------------------------------------
# AITER
# ----------------------------------------------------------

"$ROCPROF_COMPUTE" profile \\
    -n aiter_pc_sampling \\
    --no-roof \\
    --experimental \\
    --pc-sampling \\
    --pc-sampling-method stochastic \\
    --pc-sampling-interval 65536 \\
    -- "$AITER_VENV/bin/python" YOUR_AITER_SCRIPT.py


# ----------------------------------------------------------
# ROCKE
# ----------------------------------------------------------

"$ROCPROF_COMPUTE" profile \\
    -n rocke_pc_sampling \\
    --no-roof \\
    --experimental \\
    --pc-sampling \\
    --pc-sampling-method stochastic \\
    --pc-sampling-interval 65536 \\
    -- "$ROCKE_VENV/bin/python" YOUR_ROCKE_SCRIPT.py

EOF


echo
echo "============================================================"
echo "16. rocprof-compute PC sampling analysis example"
echo "============================================================"

cat <<EOF

# Replace WORKLOAD_DIR with the generated directory.

"$ROCPROF_COMPUTE" analyze \\
    -p WORKLOAD_DIR \\
    -k 0 \\
    --pc-sampling-sorting-type count

EOF


echo
echo "============================================================"
echo "17. What stochastic PC sampling gives you"
echo "============================================================"

cat <<'EOF'

The stochastic PC-sampling output can tell you:

  Wave_Issued_Instruction

  Instruction_Type

  Stall_Reason

Examples of stall reasons include:

  WAITCNT
  BARRIER_WAIT
  ALU_DEPENDENCY
  ARBITER_NOT_WIN
  ARBITER_WIN_EX_STALL

The detailed stochastic hardware snapshot can also expose
execution-pipeline arbiter state, including pipeline stalls such as:

  LDS
  VMEM/TEX
  MFMA / matrix
  VALU
  scalar

This is the data we want for comparing ROCKE vs AITER's
hardware-pipeline stall behavior.

EOF


echo
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"

echo
echo "Use this rocprofv3 for PC sampling:"
echo
echo "  $ROCPROF"
echo
echo "Launch workloads using their OWN Python environments:"
echo
echo "  AITER: $AITER_VENV/bin/python"
echo "  ROCKE: $ROCKE_VENV/bin/python"

echo
echo "For MI300X / gfx942:"
echo
echo "  method   = stochastic"
echo "  unit     = cycles"
echo "  interval = 65536"
echo
echo "============================================================"