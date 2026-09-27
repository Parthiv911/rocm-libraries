#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"

AITER_ENV="$ROOT/env/aiter_env.sh"
ROCKE_ENV="$ROOT/env/rocke_env.sh"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BENCH_AITER="$HERE/bench_aiter.py"
BENCH_ROCKE="$HERE/bench_rocke.py"

# rocprof-compute frontend
ROCPROF_COMPUTE="$ROOT/venvs/rocprof/bin/rocprof-compute"

# IMPORTANT:
# Each workload uses rocprofv3 from its OWN ROCm 10 environment.
AITER_ROCPROF="$ROOT/venvs/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$ROOT/venvs/rocke/bin/rocprofv3"

ROCKE_PYTHON="$ROOT/venvs/rocke/bin/python"

OUT="$ROOT/profiles/rocprof_compute_d128_bf16_causal"

AITER_NAME="aiter_d128_bf16_causal"
ROCKE_NAME="rocke_d128_bf16_causal"

echo "============================================================"
echo "configuration"
echo "============================================================"
echo "rocprof-compute : $ROCPROF_COMPUTE"
echo "AITER rocprofv3 : $AITER_ROCPROF"
echo "ROCKE rocprofv3 : $ROCKE_ROCPROF"
echo "ROCKE python    : $ROCKE_PYTHON"
echo "output          : $OUT"
echo

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------

[[ -x "$ROCPROF_COMPUTE" ]] || {
    echo "ERROR: rocprof-compute not executable: $ROCPROF_COMPUTE"
    exit 1
}

[[ -x "$AITER_ROCPROF" ]] || {
    echo "ERROR: AITER rocprofv3 missing: $AITER_ROCPROF"
    exit 1
}

[[ -x "$ROCKE_ROCPROF" ]] || {
    echo "ERROR: ROCKE rocprofv3 missing: $ROCKE_ROCPROF"
    exit 1
}

[[ -x "$ROCKE_PYTHON" ]] || {
    echo "ERROR: ROCKE python missing: $ROCKE_PYTHON"
    exit 1
}

[[ -f "$AITER_ENV" ]] || {
    echo "ERROR: missing $AITER_ENV"
    exit 1
}

[[ -f "$ROCKE_ENV" ]] || {
    echo "ERROR: missing $ROCKE_ENV"
    exit 1
}

[[ -f "$BENCH_AITER" ]] || {
    echo "ERROR: missing $BENCH_AITER"
    exit 1
}

[[ -f "$BENCH_ROCKE" ]] || {
    echo "ERROR: missing $BENCH_ROCKE"
    exit 1
}

echo "============================================================"
echo "versions"
echo "============================================================"

"$ROCPROF_COMPUTE" --version

echo
echo "--- AITER rocprofv3 ---"
"$AITER_ROCPROF" --version

echo
echo "--- ROCKE rocprofv3 ---"
"$ROCKE_ROCPROF" --version

mkdir -p "$OUT"

# --------------------------------------------------------------------------
# Make clean workload wrappers.
#
# These wrappers source the workload-specific ROCm environment and then exec
# directly into the benchmark process.
# --------------------------------------------------------------------------

RUN_AITER="$OUT/run_aiter_profile.sh"
RUN_ROCKE="$OUT/run_rocke_profile.sh"

cat > "$RUN_AITER" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

source "$AITER_ENV"

export AITER_WARMUP=0
export AITER_REPEAT=1

exec python3 "$BENCH_AITER"
EOF

cat > "$RUN_ROCKE" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

source "$ROCKE_ENV"

cd "$PROVIDER"

exec "$ROCKE_PYTHON" "$BENCH_ROCKE"
EOF

chmod +x "$RUN_AITER" "$RUN_ROCKE"

# --------------------------------------------------------------------------
# Prewarm outside profiler
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "AITER prewarm"
echo "============================================================"

bash -lc "
source '$AITER_ENV'
AITER_WARMUP=10 AITER_REPEAT=1 python3 '$BENCH_AITER'
"

echo
echo "============================================================"
echo "ROCKE prewarm"
echo "============================================================"

bash -lc "
source '$ROCKE_ENV'
cd '$PROVIDER'
'$ROCKE_PYTHON' '$BENCH_ROCKE'
"

cd "$OUT"

# --------------------------------------------------------------------------
# AITER
#
# NOTE:
# ROCPROF is deliberately scoped to this ONE command.
#
# DO NOT use:
#   /root/gpu-bench/venvs/rocprof/bin/rocprofv3
#
# We already proved AITER's own ROCm-10 rocprofv3 collects counters correctly.
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "PROFILE AITER"
echo "============================================================"

ROCPROF="$AITER_ROCPROF" \
"$ROCPROF_COMPUTE" profile \
    --overwrite \
    -n "$AITER_NAME" \
    -b 2 3 7 10 12 16 17 \
    -k fmha_fwd_hd128_bf16_causal_rtz \
    -- \
    "$RUN_AITER"

# --------------------------------------------------------------------------
# ROCKE
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "PROFILE ROCKE"
echo "============================================================"

ROCPROF="$ROCKE_ROCPROF" \
"$ROCPROF_COMPUTE" profile \
    --overwrite \
    -n "$ROCKE_NAME" \
    -b 2 3 7 10 12 16 17 \
    -k rocke_attention_dense \
    -- \
    "$RUN_ROCKE"

# --------------------------------------------------------------------------
# Locate generated workload directories
# --------------------------------------------------------------------------

AITER_RUN="$(
    find "$OUT/workloads/$AITER_NAME" \
        -mindepth 1 -maxdepth 1 -type d -print -quit
)"

ROCKE_RUN="$(
    find "$OUT/workloads/$ROCKE_NAME" \
        -mindepth 1 -maxdepth 1 -type d -print -quit
)"

[[ -n "$AITER_RUN" ]] || {
    echo "ERROR: couldn't locate AITER workload output"
    exit 1
}

[[ -n "$ROCKE_RUN" ]] || {
    echo "ERROR: couldn't locate ROCKE workload output"
    exit 1
}

echo
echo "============================================================"
echo "PROFILE OUTPUT"
echo "============================================================"

echo "AITER:"
echo "  $AITER_RUN"

echo
echo "ROCKE:"
echo "  $ROCKE_RUN"

# --------------------------------------------------------------------------
# Generate useful text analysis
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "MEMORY ANALYSIS"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" \
    -b 3 10.3 12 16 17 \
    > "$OUT/aiter_memory.txt"

"$ROCPROF_COMPUTE" analyze \
    -p "$ROCKE_RUN" \
    -b 3 10.3 12 16 17 \
    > "$OUT/rocke_memory.txt"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" "$ROCKE_RUN" \
    -b 10.3 12 16 17 \
    > "$OUT/compare_memory.txt"

# --------------------------------------------------------------------------
# BF16 ROOFLINE
#
# Profile mode above already collected roofline.csv because --no-roof is gone.
#
# The datatype is selected at ANALYZE time.
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "BF16 ROOFLINE"
echo "============================================================"

echo "Generating AITER BF16 roofline..."
"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" \
    -b 4 \
    --roofline-data-type BF16 \
    || echo "WARNING: AITER roofline analysis failed"

echo
echo "Generating ROCKE BF16 roofline..."
"$ROCPROF_COMPUTE" analyze \
    -p "$ROCKE_RUN" \
    -b 4 \
    --roofline-data-type BF16 \
    || echo "WARNING: ROCKE roofline analysis failed"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "AITER:"
echo "  $AITER_RUN"

echo
echo "ROCKE:"
echo "  $ROCKE_RUN"

echo
echo "Text reports:"
echo "  $OUT/aiter_memory.txt"
echo "  $OUT/rocke_memory.txt"
echo "  $OUT/compare_memory.txt"

echo
echo "BF16 roofline:"
echo "  $ROCPROF_COMPUTE analyze -p \"$AITER_RUN\" -b 4 --roofline-data-type BF16"
echo "  $ROCPROF_COMPUTE analyze -p \"$ROCKE_RUN\" -b 4 --roofline-data-type BF16"

echo
echo "GUI:"
echo "  $ROCPROF_COMPUTE analyze -p \"$AITER_RUN\" \"$ROCKE_RUN\" --experimental --gui"