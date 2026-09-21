#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

LIB="$ROCM_REPO/dnn-providers/hip-kernel-provider/rocke/library"
SOURCE="$LIB/kernels/gfx942/attention_dense.py"

PY="$SCRIPT_DIR/bench_rocke_once.py"
OUT="$SCRIPT_DIR/results"

ROOT="${GPU_BENCH_ROOT:-/root/gpu-bench}"
VENV="$ROOT/venvs/rocke"

ROUNDS="${1:-5}"

CSV="$OUT/results.csv"
LOG="$OUT/all_runs.log"
VALIDATION_LOG="$OUT/validation.txt"

mkdir -p "$OUT"

echo "B,S,HQ,HKV,D,round,order,variant,cfvst,ms" > "$CSV"
: > "$LOG"
: > "$VALIDATION_LOG"


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source "$VENV/bin/activate"

SITE="$VENV/lib/python3.12/site-packages"
CORE="$SITE/_rocm_sdk_core"

export ROCKE_BACKEND=python
export ROCKE_LLVM_FLAVOR=llvm23
export ROCKE_COMGR_LIB="$CORE/lib/libamd_comgr.so.3"
export ROCKE_HIP_LIB="$CORE/lib/libamdhip64.so.7"
export LD_LIBRARY_PATH="$CORE/lib"

unset LD_PRELOAD

export PYTHONPATH="$LIB${PYTHONPATH:+:$PYTHONPATH}"


# ------------------------------------------------------------
# Preserve the original source file.
#
# This guarantees that even if validation or benchmarking fails,
# attention_dense.py is restored to its original state.
# ------------------------------------------------------------

SOURCE_BACKUP="$(mktemp)"
cp "$SOURCE" "$SOURCE_BACKUP"

cleanup() {
    cp "$SOURCE_BACKUP" "$SOURCE"
    rm -f "$SOURCE_BACKUP"
}

trap cleanup EXIT


# ------------------------------------------------------------
# CFVST policy variants
# ------------------------------------------------------------

set_off() {
    python - "$SOURCE" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

off = 'return _rows_per_instr(head_size) == 1 and dtype == "fp16"'
on  = 'return _rows_per_instr(head_size) == 1'

if off in s:
    pass
elif on in s:
    s = s.replace(on, off, 1)
    p.write_text(s)
else:
    raise SystemExit(
        "ERROR: could not find CFVST policy return line"
    )
PY
}


set_on() {
    python - "$SOURCE" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

off = 'return _rows_per_instr(head_size) == 1 and dtype == "fp16"'
on  = 'return _rows_per_instr(head_size) == 1'

if off in s:
    s = s.replace(off, on, 1)
    p.write_text(s)
elif on in s:
    pass
else:
    raise SystemExit(
        "ERROR: could not find CFVST policy return line"
    )
PY
}


# ------------------------------------------------------------
# Numerical validation for one variant / shape
# ------------------------------------------------------------

validate_one() {
    local B="$1"
    local S="$2"
    local HQ="$3"
    local HKV="$4"
    local D="$5"
    local variant="$6"
    local expected="$7"

    local shape="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    local tmp="$OUT/${shape}_${variant}_validation.txt"

    echo
    echo "------------------------------------------------"
    echo "VALIDATE: $variant"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "------------------------------------------------"

    B="$B" \
    S="$S" \
    HQ="$HQ" \
    HKV="$HKV" \
    D="$D" \
    VALIDATE=1 \
        python "$PY" \
        | tee "$tmp" \
        | tee -a "$VALIDATION_LOG" \
        | tee -a "$LOG"

    local state

    state="$(awk '/^cfvst:/ {print $2}' "$tmp" | tail -1)"

    if [[ "$state" != "$expected" ]]; then
        echo "ERROR: expected cfvst=$expected, got $state"
        exit 1
    fi

    if ! grep -q '^VALIDATION=PASS' "$tmp"; then
        echo "ERROR: numerical validation did not pass"
        exit 1
    fi
}


# ------------------------------------------------------------
# Performance measurement for one variant / shape
# ------------------------------------------------------------

run_one() {
    local B="$1"
    local S="$2"
    local HQ="$3"
    local HKV="$4"
    local D="$5"
    local round="$6"
    local order="$7"
    local variant="$8"
    local expected="$9"

    local shape="b${B}_s${S}_hq${HQ}_hkv${HKV}_d${D}"
    local tmp="$OUT/${shape}_${variant}_${round}.txt"

    echo
    echo "================================================"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "ROUND $round ($order) : $variant"
    echo "================================================"

    B="$B" \
    S="$S" \
    HQ="$HQ" \
    HKV="$HKV" \
    D="$D" \
        python "$PY" \
        | tee "$tmp" \
        | tee -a "$LOG"

    local state
    local ms

    state="$(awk '/^cfvst:/ {print $2}' "$tmp" | tail -1)"
    ms="$(awk -F= '/^RESULT_MS=/ {print $2}' "$tmp" | tail -1)"

    if [[ "$state" != "$expected" ]]; then
        echo "ERROR: expected cfvst=$expected, got $state"
        exit 1
    fi

    if [[ -z "$ms" ]]; then
        echo "ERROR: RESULT_MS missing"
        exit 1
    fi

    echo \
"$B,$S,$HQ,$HKV,$D,$round,$order,$variant,$state,$ms" \
        >> "$CSV"
}


# ------------------------------------------------------------
# BF16 D128 test shapes
#
# Format:
#
# B S HQ HKV D
# ------------------------------------------------------------

SHAPES=(
    "1 4096 32 8 128"
    "1 4096 32 16 128"

    "1 8192 32 8 128"
    "1 8192 32 16 128"

    "1 16384 32 8 128"

    "16 4096 32 8 128"
    "16 4096 32 16 128"

    "16 8192 32 8 128"
)


# ------------------------------------------------------------
# Run every shape
# ------------------------------------------------------------

for shape in "${SHAPES[@]}"; do

    read -r B S HQ HKV D <<< "$shape"

    echo
    echo
    echo "################################################"
    echo "SHAPE"
    echo "B=$B S=$S HQ=$HQ HKV=$HKV D=$D"
    echo "################################################"


    # --------------------------------------------------------
    # Numerical correctness
    # --------------------------------------------------------

    echo
    echo "NUMERICAL VALIDATION"

    set_off
    validate_one \
        "$B" "$S" "$HQ" "$HKV" "$D" \
        "A_without_cfvst" \
        "False"

    set_on
    validate_one \
        "$B" "$S" "$HQ" "$HKV" "$D" \
        "B_with_cfvst" \
        "True"

    echo
    echo "Both variants passed numerical validation."


    # --------------------------------------------------------
    # Performance
    #
    # Odd rounds  : A -> B
    # Even rounds : B -> A
    #
    # This prevents a systematic A-first ordering bias.
    # --------------------------------------------------------

    for ((i=1; i<=ROUNDS; i++)); do

        if (( i % 2 == 1 )); then

            order="AB"

            set_off
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "A_without_cfvst" \
                "False"

            set_on
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "B_with_cfvst" \
                "True"

        else

            order="BA"

            set_on
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "B_with_cfvst" \
                "True"

            set_off
            run_one \
                "$B" "$S" "$HQ" "$HKV" "$D" \
                "$i" "$order" \
                "A_without_cfvst" \
                "False"

        fi

    done

done


# ------------------------------------------------------------
# Raw results
# ------------------------------------------------------------

echo
echo
echo "================================================"
echo "RESULTS"
echo "================================================"

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"


# ------------------------------------------------------------
# Per-shape summary
# ------------------------------------------------------------

echo
echo "================================================"
echo "SUMMARY"
echo "================================================"

python - "$CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict


data = defaultdict(
    lambda: {
        "A_without_cfvst": [],
        "B_with_cfvst": [],
    }
)


with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):

        key = (
            int(row["B"]),
            int(row["S"]),
            int(row["HQ"]),
            int(row["HKV"]),
            int(row["D"]),
        )

        data[key][row["variant"]].append(
            float(row["ms"])
        )


print(
    f"{'B':>4} "
    f"{'S':>7} "
    f"{'HQ':>4} "
    f"{'HKV':>4} "
    f"{'D':>4} "
    f"{'A median ms':>13} "
    f"{'B median ms':>13} "
    f"{'Reduction':>11} "
    f"{'Speedup':>9}"
)

print("-" * 82)


for key in sorted(data):

    B, S, HQ, HKV, D = key

    a = data[key]["A_without_cfvst"]
    b = data[key]["B_with_cfvst"]

    if not a or not b:
        print(
            f"ERROR: missing A or B data for "
            f"B={B} S={S} HQ={HQ} HKV={HKV} D={D}"
        )
        continue

    ma = statistics.median(a)
    mb = statistics.median(b)

    reduction = (ma - mb) / ma * 100.0
    speedup = ma / mb

    print(
        f"{B:4d} "
        f"{S:7d} "
        f"{HQ:4d} "
        f"{HKV:4d} "
        f"{D:4d} "
        f"{ma:13.6f} "
        f"{mb:13.6f} "
        f"{reduction:10.2f}% "
        f"{speedup:8.4f}x"
    )


print()
print("Detailed measurements:")

for key in sorted(data):

    B, S, HQ, HKV, D = key

    a = data[key]["A_without_cfvst"]
    b = data[key]["B_with_cfvst"]

    print()
    print(
        f"B={B} S={S} HQ={HQ} HKV={HKV} D={D}"
    )

    print(
        "  A:",
        ", ".join(f"{x:.6f}" for x in a),
    )

    print(
        "  B:",
        ", ".join(f"{x:.6f}" for x in b),
    )
PY


echo
echo "CSV:        $CSV"
echo "Log:        $LOG"
echo "Validation: $VALIDATION_LOG"
echo
echo "Original attention_dense.py restored."