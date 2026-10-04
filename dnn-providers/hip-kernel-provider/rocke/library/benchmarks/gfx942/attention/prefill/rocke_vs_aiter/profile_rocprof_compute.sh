#!/usr/bin/env bash
set -Eeuo pipefail

export ROCM_VER=10.0.0

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

# Each workload uses rocprofv3 from its own ROCm 10 environment.
AITER_ROCPROF="$ROOT/venvs/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$ROOT/venvs/rocke/bin/rocprofv3"

ROCKE_PYTHON="$ROOT/venvs/rocke/bin/python"

OUT="$ROOT/profiles/rocprof_compute_d128_bf16_causal"

AITER_NAME="aiter_d128_bf16_causal"
ROCKE_NAME="rocke_d128_bf16_causal"

# Target kernels.
#
# bench_aiter.py uses:
#   -v3_bf16_cvt=0
#
# which selects RTNE:
#   fmha_fwd_hd128_bf16_causal_rtne
AITER_KERNEL="fmha_fwd_hd128_bf16_causal_rtne"
ROCKE_KERNEL="rocke_attention_dense"

# --------------------------------------------------------------------------
# PC sampling
#
# MI300X / gfx942 supports stochastic hardware PC sampling.
#
# The default stochastic interval is 1048576 cycles. These attention
# kernels are sub-millisecond, so use the minimum supported interval
# (65536 cycles) to get substantially better sample density.
# --------------------------------------------------------------------------

PC_SAMPLING_METHOD="stochastic"
PC_SAMPLING_INTERVAL="65536"

echo "============================================================"
echo "configuration"
echo "============================================================"
echo "rocprof-compute : $ROCPROF_COMPUTE"
echo "AITER rocprofv3 : $AITER_ROCPROF"
echo "ROCKE rocprofv3 : $ROCKE_ROCPROF"
echo "ROCKE python    : $ROCKE_PYTHON"
echo "AITER kernel    : $AITER_KERNEL"
echo "ROCKE kernel    : $ROCKE_KERNEL"
echo "output          : $OUT"
echo "PC method       : $PC_SAMPLING_METHOD"
echo "PC interval     : $PC_SAMPLING_INTERVAL cycles"
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
# Workload wrappers
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

# ==========================================================================
# AITER
#
# IMPORTANT:
#
# No -b/--block option is used.
#
# Therefore rocprof-compute collects ALL standard available analysis
# counters rather than only selected blocks.
#
# PC sampling is additionally enabled explicitly because it is not part
# of normal counter collection.
#
# Roofline is also collected automatically because --no-roof is NOT used.
# ==========================================================================

echo
echo "============================================================"
echo "PROFILE AITER -- FULL COUNTERS + ROOFLINE + PC SAMPLING"
echo "============================================================"

ROCPROF="$AITER_ROCPROF" \
"$ROCPROF_COMPUTE" profile \
    --overwrite \
    --experimental \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval 65536 \
    -n "$AITER_NAME" \
    -b 2 3 4 5 6 7 10 11 12 13 14 15 16 17 18 21 \
    -k "$AITER_KERNEL" \
    -- \
    "$RUN_AITER"

# ==========================================================================
# ROCKE
# ==========================================================================

echo
echo "============================================================"
echo "PROFILE ROCKE -- FULL COUNTERS + ROOFLINE + PC SAMPLING"
echo "============================================================"

ROCPROF="$ROCKE_ROCPROF" \
"$ROCPROF_COMPUTE" profile \
    --overwrite \
    --experimental \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval 65536 \
    -n "$ROCKE_NAME" \
    -b 2 3 4 5 6 7 10 11 12 13 14 15 16 17 18 21 \
    -k "$ROCKE_KERNEL" \
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
# Verify important collected files
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "COLLECTION SANITY CHECK"
echo "============================================================"

echo
echo "--- AITER files ---"
find "$AITER_RUN" -maxdepth 2 -type f | sort

echo
echo "--- ROCKE files ---"
find "$ROCKE_RUN" -maxdepth 2 -type f | sort

echo
echo "--- Roofline files ---"
find "$AITER_RUN" "$ROCKE_RUN" \
    -name 'roofline.csv' \
    -print

echo
echo "--- PC sampling files ---"
find "$AITER_RUN" "$ROCKE_RUN" \
    \( -iname '*pc*sampling*' -o -iname '*sampling*' \) \
    -print || true

# ==========================================================================
# FULL INDIVIDUAL REPORTS
#
# No -b filter.
#
# This prints every analysis block for which data was collected.
#
# --pc-sampling-rows 0:
#     show ALL sampled PC/ISA rows instead of the default top 10.
#
# --pc-sampling-sorting-type count:
#     hottest / most frequently sampled instructions first.
#
# BF16 is selected for roofline visualization.
# ==========================================================================

echo
echo "============================================================"
echo "FULL AITER ANALYSIS"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" \
    --roofline-data-type BF16 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 0 \
    > "$OUT/aiter_full_report.txt"

echo
echo "============================================================"
echo "FULL ROCKE ANALYSIS"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$ROCKE_RUN" \
    --roofline-data-type BF16 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 0 \
    > "$OUT/rocke_full_report.txt"

# ==========================================================================
# FULL SIDE-BY-SIDE COMPARISON
# ==========================================================================

echo
echo "============================================================"
echo "FULL AITER vs ROCKE COMPARISON"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" "$ROCKE_RUN" \
    --roofline-data-type BF16 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 0 \
    > "$OUT/compare_full_report.txt"

# --------------------------------------------------------------------------
# Separate PC-sampling-only reports
#
# Block 21 = PC Sampling.
# Useful because the full report is very large.
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "PC SAMPLING REPORTS"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" \
    -b 21 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 0 \
    > "$OUT/aiter_pc_sampling.txt"

"$ROCPROF_COMPUTE" analyze \
    -p "$ROCKE_RUN" \
    -b 21 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 0 \
    > "$OUT/rocke_pc_sampling.txt"

# --------------------------------------------------------------------------
# Memory-focused reports retained as convenience reports.
#
# These are subsets only; the full reports above contain everything.
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "MEMORY-FOCUSED REPORTS"
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
# Roofline-only analysis
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "BF16 ROOFLINE"
echo "============================================================"

"$ROCPROF_COMPUTE" analyze \
    -p "$AITER_RUN" \
    -b 4 \
    --roofline-data-type BF16 \
    || echo "WARNING: AITER roofline analysis failed"

"$ROCPROF_COMPUTE" analyze \
    -p "$ROCKE_RUN" \
    -b 4 \
    --roofline-data-type BF16 \
    || echo "WARNING: ROCKE roofline analysis failed"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo
echo "AITER workload:"
echo "  $AITER_RUN"

echo
echo "ROCKE workload:"
echo "  $ROCKE_RUN"

echo
echo "Full reports:"
echo "  $OUT/aiter_full_report.txt"
echo "  $OUT/rocke_full_report.txt"
echo "  $OUT/compare_full_report.txt"

echo
echo "PC sampling reports:"
echo "  $OUT/aiter_pc_sampling.txt"
echo "  $OUT/rocke_pc_sampling.txt"

echo
echo "Memory reports:"
echo "  $OUT/aiter_memory.txt"
echo "  $OUT/rocke_memory.txt"
echo "  $OUT/compare_memory.txt"

echo
echo "AITER GUI:"
echo "  $ROCPROF_COMPUTE analyze -p \"$AITER_RUN\" --experimental --gui 8050"

echo
echo "ROCKE GUI:"
echo "  $ROCPROF_COMPUTE analyze -p \"$ROCKE_RUN\" --experimental --gui 8051"

echo
echo "Combined GUI:"
echo "  $ROCPROF_COMPUTE analyze -p \"$AITER_RUN\" \"$ROCKE_RUN\" --experimental --gui"