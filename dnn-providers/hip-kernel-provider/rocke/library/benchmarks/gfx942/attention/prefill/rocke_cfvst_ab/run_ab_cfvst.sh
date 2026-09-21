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

mkdir -p "$OUT"

echo "round,variant,cfvst,ms" > "$CSV"
: > "$LOG"

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

OFF='return _rows_per_instr(head_size) == 1 and dtype == "fp16"'
ON='return _rows_per_instr(head_size) == 1'

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
    raise SystemExit("ERROR: could not find CFVST policy return line")
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
    raise SystemExit("ERROR: could not find CFVST policy return line")
PY
}

run_one() {
    local round="$1"
    local variant="$2"
    local expected="$3"

    local tmp="$OUT/${variant}_${round}.txt"

    echo
    echo "================================================"
    echo "ROUND $round : $variant"
    echo "================================================"

    python "$PY" | tee "$tmp" | tee -a "$LOG"

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

    echo "$round,$variant,$state,$ms" >> "$CSV"
}

echo
echo "================================================"
echo "NUMERICAL VALIDATION"
echo "================================================"

echo
echo "Validating A_without_cfvst..."
set_off
VALIDATE=1 python "$PY"

echo
echo "Validating B_with_cfvst..."
set_on
VALIDATE=1 python "$PY"

echo
echo "Both variants passed numerical validation."
echo

for ((i=1; i<=ROUNDS; i++)); do
    set_off
    run_one "$i" "A_without_cfvst" "False"

    set_on
    run_one "$i" "B_with_cfvst" "True"
done

# Leave source with CFVST enabled.
set_on

echo
echo "================================================"
echo "RESULTS"
echo "================================================"

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"

echo
echo "================================================"
echo "SUMMARY"
echo "================================================"

python - "$CSV" <<'PY'
import csv
import statistics
import sys

a = []
b = []

with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        x = float(r["ms"])

        if r["variant"] == "A_without_cfvst":
            a.append(x)
        elif r["variant"] == "B_with_cfvst":
            b.append(x)

ma = statistics.median(a)
mb = statistics.median(b)

print("Without CFVST:")
for i, x in enumerate(a, 1):
    print(f"  A{i}: {x:.6f} ms")

print()
print("With CFVST:")
for i, x in enumerate(b, 1):
    print(f"  B{i}: {x:.6f} ms")

print()
print(f"Median without CFVST : {ma:.6f} ms")
print(f"Median with CFVST    : {mb:.6f} ms")
print(f"Latency reduction    : {(ma - mb) / ma * 100:.2f}%")
print(f"Speedup              : {ma / mb:.4f}x")
PY

echo
echo "CSV: $CSV"
echo "Log: $LOG"
echo "Source left with CFVST enabled."
