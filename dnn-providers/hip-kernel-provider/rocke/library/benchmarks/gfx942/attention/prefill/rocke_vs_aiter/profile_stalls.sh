#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/gpu-bench
REPO="$ROOT/rocm-libraries"
AITER="$ROOT/aiter"
VENV="$ROOT/venvs"
ENV="$ROOT/env"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HERE/profiling/first_shape"
ROCKE_HELPER="$BASE/rocke_first_shape.py"

# Do not overwrite the counters you already collected.
OUT="$BASE/hardware_counters_stalls"

AITER_ROCPROF="$VENV/aiter/bin/rocprofv3"
ROCKE_ROCPROF="$VENV/rocke/bin/rocprofv3"

AITER_AVAIL="$VENV/aiter/bin/rocprofv3-avail"
ROCKE_AVAIL="$VENV/rocke/bin/rocprofv3-avail"

[ -x "$AITER_ROCPROF" ] || {
  echo "missing $AITER_ROCPROF"
  exit 1
}

[ -x "$ROCKE_ROCPROF" ] || {
  echo "missing $ROCKE_ROCPROF"
  exit 1
}

[ -x "$AITER_AVAIL" ] || {
  echo "missing $AITER_AVAIL"
  exit 1
}

[ -x "$ROCKE_AVAIL" ] || {
  echo "missing $ROCKE_AVAIL"
  exit 1
}

[ -x "$AITER/op_tests/cpp/mha/fwd.exe" ] || {
  echo "missing AITER fwd.exe"
  exit 1
}

[ -f "$ROCKE_HELPER" ] || {
  echo "missing $ROCKE_HELPER"
  echo "run the single-dispatch profile.sh first"
  exit 1
}

mkdir -p "$OUT/aiter" "$OUT/rocke"


# ============================================================
# Candidate counters
#
# These target the missing questions from the first profile:
#
#   1. How much time are waves waiting?
#   2. What kind of instruction is occupying issue slots?
#   3. How much VMEM/LDS concurrency exists?
#   4. Are LDS stalls actually blocking ALU progress?
#   5. Is L2 / memory-system pressure the bottleneck?
#
# Unsupported counters are automatically skipped.
# ============================================================

CANDIDATES=(
  # Normalization / execution
  SQ_WAVE_CYCLES
  SQ_BUSY_CU_CYCLES

  # Overall waiting
  SQ_WAIT_ANY
  SQ_WAIT_INST_ANY

  # Instruction issue/activity
  SQ_ACTIVE_INST_ANY
  SQ_ACTIVE_INST_VMEM
  SQ_ACTIVE_INST_LDS
  SQ_ACTIVE_INST_VALU
  SQ_ACTIVE_INST_FLAT
  SQ_ACTIVE_INST_SCA
  SQ_ACTIVE_INST_MISC

  # Number of operations in flight
  SQ_INST_LEVEL_VMEM
  SQ_INST_LEVEL_LDS

  # Extra instruction overhead
  SQ_INSTS_BRANCH
  SQ_INSTS_FLAT

  # LDS-derived bottleneck metric
  ALUStalledByLDS

  # Memory-system derived metrics
  MemUnitBusy
  MemUnitStalled
  WriteUnitStalled
  L2CacheHit

  # Compute utilization
  VALUBusy
  VALUUtilization

  # Additional L2/TCC pressure counters
  TCC_BUSY_sum
  TCC_TAG_STALL_sum
)


# ============================================================
# Discover counters actually available in each environment
# ============================================================

echo "============================================================"
echo "QUERYING AVAILABLE COUNTERS"
echo "============================================================"

"$AITER_AVAIL" info --pmc > "$OUT/aiter_available_counters.txt"
"$ROCKE_AVAIL" info --pmc > "$OUT/rocke_available_counters.txt"


extract_counter_names() {
  local input="$1"
  local output="$2"

  awk -F: '
    /Counter_Name/ {
      name=$2
      gsub(/^[ \t]+/, "", name)
      gsub(/[ \t]+$/, "", name)

      if (name != "")
        print name
    }
  ' "$input" | sort -u > "$output"
}


AITER_NAMES="$OUT/aiter_counter_names.txt"
ROCKE_NAMES="$OUT/rocke_counter_names.txt"

extract_counter_names \
  "$OUT/aiter_available_counters.txt" \
  "$AITER_NAMES"

extract_counter_names \
  "$OUT/rocke_available_counters.txt" \
  "$ROCKE_NAMES"


# ============================================================
# Keep only counters supported by BOTH environments
# ============================================================

COMMON="$OUT/common_counters.txt"
SKIPPED="$OUT/skipped_counters.txt"

: > "$COMMON"
: > "$SKIPPED"

for counter in "${CANDIDATES[@]}"; do
  if grep -Fxq "$counter" "$AITER_NAMES" &&
     grep -Fxq "$counter" "$ROCKE_NAMES"; then

    echo "$counter" >> "$COMMON"

  else
    echo "$counter" >> "$SKIPPED"
  fi
done


echo
echo "Counters that will be collected:"
cat "$COMMON"

echo
echo "Unsupported / skipped:"
cat "$SKIPPED"


mapfile -t COUNTERS < "$COMMON"

if [ "${#COUNTERS[@]}" -eq 0 ]; then
  echo "ERROR: none of the requested counters are supported"
  exit 1
fi


# ============================================================
# Generate YAML
# ============================================================

write_yaml() {
  local path="$1"
  local regex="$2"

  {
    echo "jobs:"

    for counter in "${COUNTERS[@]}"; do
      echo "  - pmc: [\"$counter\"]"
      echo "    kernel_include_regex: \"$regex\""
    done

  } > "$path"
}


write_map() {
  local path="$1"

  {
    printf "pass,counter\n"

    local i=1

    for counter in "${COUNTERS[@]}"; do
      printf "pass_%d,%s\n" "$i" "$counter"
      i=$((i + 1))
    done

  } > "$path"
}


AITER_YAML="$OUT/aiter_counters.yaml"
ROCKE_YAML="$OUT/rocke_counters.yaml"

write_yaml \
  "$AITER_YAML" \
  "fmha_fwd_hd128_bf16_causal_rtz"

write_yaml \
  "$ROCKE_YAML" \
  "rocke_attention_dense"

write_map "$OUT/pass_map.csv"


rm -rf \
  "$OUT/aiter"/pass_* \
  "$OUT/rocke"/pass_*


# ============================================================
# AITER
# ============================================================

echo
echo "============================================================"
echo "AITER STALL / ISSUE COUNTERS"
echo "============================================================"

bash -lc "
source '$ENV/aiter_env.sh'

cd '$AITER/op_tests/cpp/mha'

'$AITER_ROCPROF' \
  -i '$AITER_YAML' \
  --output-format csv \
  --output-directory '$OUT/aiter' \
  -- \
  ./fwd.exe \
    -prec=bf16 \
    -b=1 \
    -h=32 \
    -h_k=8 \
    -d=128 \
    -d_v=128 \
    -s=4096 \
    -s_k=4096 \
    -iperm=0 \
    -operm=0 \
    -mask=1 \
    -lse=0 \
    -fwd_v3=1 \
    -v3_bf16_cvt=2 \
    -mode=0 \
    -timer=gpu \
    -kname=1 \
    -v=0 \
    -warmup=0 \
    -repeat=1
" 2>&1 | tee "$OUT/aiter/rocprofv3_counters.log"


# ============================================================
# ROCKE
# ============================================================

echo
echo "============================================================"
echo "ROCKE STALL / ISSUE COUNTERS"
echo "============================================================"

bash -lc "
source '$ENV/rocke_env.sh'

cd '$REPO/dnn-providers/hip-kernel-provider'

'$ROCKE_ROCPROF' \
  -i '$ROCKE_YAML' \
  --output-format csv \
  --output-directory '$OUT/rocke' \
  -- \
  '$VENV/rocke/bin/python' \
  '$ROCKE_HELPER'
" 2>&1 | tee "$OUT/rocke/rocprofv3_counters.log"


# ============================================================
# README
# ============================================================

cat > "$OUT/README.txt" <<EOF
Additional stall / issue analysis for first attention shape.

Shape:
B=1
S=4096
Hq=32
Hkv=8
D=128
dtype=BF16
causal=true
layout=BSHD

Purpose:

This counter set supplements hardware_counters/.

The original counter set answers:

- MFMA utilization
- LDS bank conflicts
- VMEM latency
- instruction counts
- L2 hits/misses
- DRAM traffic

This set attempts to answer:

- how many wave cycles are spent waiting
- what instruction pipelines are active
- VMEM instructions in flight
- LDS instructions in flight
- whether LDS stalls block ALUs
- whether the memory subsystem is busy/stalled
- whether extra branch/flat work exists

Each PMC is collected in its own rocprofv3 job/pass.

Counters unsupported by the installed ROCm/gfx942 stack are skipped
automatically.

See:

common_counters.txt
skipped_counters.txt
pass_map.csv
EOF


# ============================================================
# Consolidate
# ============================================================

echo
echo "============================================================"
echo "CONSOLIDATING"
echo "============================================================"

python3 - "$OUT" <<'PY'
import csv
import glob
import math
import os
import sys


root = sys.argv[1]


# ------------------------------------------------------------
# pass_N -> counter
# ------------------------------------------------------------

pass_map = {}

with open(os.path.join(root, "pass_map.csv"), newline="") as f:
    for row in csv.DictReader(f):
        pass_map[row["pass"]] = row["counter"]


def pass_number(name):
    return int(name.split("_")[1])


# ------------------------------------------------------------
# Read one implementation
# ------------------------------------------------------------

def read_variant(variant):
    result = {}

    for p in sorted(pass_map, key=pass_number):

        counter = pass_map[p]

        pattern = os.path.join(
            root,
            variant,
            p,
            "*",
            "*_counter_collection.csv",
        )

        files = glob.glob(pattern)

        if len(files) != 1:
            raise SystemExit(
                f"ERROR: expected exactly one counter CSV for "
                f"{variant}/{p}; found {len(files)}"
            )

        with open(files[0], newline="") as f:
            rows = list(csv.DictReader(f))

        matches = [
            row
            for row in rows
            if row["Counter_Name"] == counter
        ]

        if len(matches) != 1:
            raise SystemExit(
                f"ERROR: expected exactly one {counter} row for "
                f"{variant}/{p}; found {len(matches)}"
            )

        result[counter] = float(matches[0]["Counter_Value"])

    return result


aiter = read_variant("aiter")
rocke = read_variant("rocke")


# ------------------------------------------------------------
# Raw comparison
# ------------------------------------------------------------

comparison_path = os.path.join(
    root,
    "counter_comparison.csv",
)

with open(comparison_path, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow([
        "Counter",
        "AITER",
        "ROCKE",
        "ROCKE_over_AITER",
    ])

    for p in sorted(pass_map, key=pass_number):

        counter = pass_map[p]

        av = aiter[counter]
        rv = rocke[counter]

        ratio = (
            rv / av
            if av != 0
            else float("nan")
        )

        w.writerow([
            counter,
            av,
            rv,
            ratio,
        ])


# ------------------------------------------------------------
# Derived normalized metrics
# ------------------------------------------------------------

def get(d, key):
    return d.get(key)


def divide(a, b):
    if a is None or b is None or b == 0:
        return None

    return a / b


derived = []


def add_metric(name, fn):
    av = fn(aiter)
    rv = fn(rocke)

    ratio = (
        divide(rv, av)
        if av not in (None, 0)
        else None
    )

    derived.append(
        (name, av, rv, ratio)
    )


# Waiting normalized by total wave cycles.
add_metric(
    "wait_any_per_wave_cycle",
    lambda x: divide(
        get(x, "SQ_WAIT_ANY"),
        get(x, "SQ_WAVE_CYCLES"),
    ),
)

add_metric(
    "wait_inst_any_per_wave_cycle",
    lambda x: divide(
        get(x, "SQ_WAIT_INST_ANY"),
        get(x, "SQ_WAVE_CYCLES"),
    ),
)


# Number of in-flight operations relative to wave execution.
add_metric(
    "vmem_inflight_per_wave_cycle",
    lambda x: divide(
        get(x, "SQ_INST_LEVEL_VMEM"),
        get(x, "SQ_WAVE_CYCLES"),
    ),
)

add_metric(
    "lds_inflight_per_wave_cycle",
    lambda x: divide(
        get(x, "SQ_INST_LEVEL_LDS"),
        get(x, "SQ_WAVE_CYCLES"),
    ),
)


# Issue activity normalized to busy-CU cycles.
for counter, label in [
    ("SQ_ACTIVE_INST_VMEM", "active_vmem_per_busy_cycle"),
    ("SQ_ACTIVE_INST_LDS", "active_lds_per_busy_cycle"),
    ("SQ_ACTIVE_INST_VALU", "active_valu_per_busy_cycle"),
    ("SQ_ACTIVE_INST_FLAT", "active_flat_per_busy_cycle"),
    ("SQ_ACTIVE_INST_SCA", "active_scalar_per_busy_cycle"),
    ("SQ_ACTIVE_INST_MISC", "active_misc_per_busy_cycle"),
]:

    add_metric(
        label,
        lambda x, c=counter: divide(
            get(x, c),
            get(x, "SQ_BUSY_CU_CYCLES"),
        ),
    )


derived_path = os.path.join(
    root,
    "derived_summary.csv",
)

with open(derived_path, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow([
        "Metric",
        "AITER",
        "ROCKE",
        "ROCKE_over_AITER",
    ])

    for row in derived:
        w.writerow(row)


# ------------------------------------------------------------
# Pretty stdout
# ------------------------------------------------------------

print()
print(
    f"{'Counter':<32} "
    f"{'AITER':>18} "
    f"{'ROCKE':>18} "
    f"{'R/A':>10}"
)

print("-" * 82)

for p in sorted(pass_map, key=pass_number):

    counter = pass_map[p]

    av = aiter[counter]
    rv = rocke[counter]

    ratio = (
        rv / av
        if av != 0
        else float("nan")
    )

    print(
        f"{counter:<32} "
        f"{av:>18.6g} "
        f"{rv:>18.6g} "
        f"{ratio:>10.3f}"
    )


print()
print("Derived normalized metrics:")
print()

print(
    f"{'Metric':<36} "
    f"{'AITER':>15} "
    f"{'ROCKE':>15} "
    f"{'R/A':>10}"
)

print("-" * 82)


for name, av, rv, ratio in derived:

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.6g}"

    print(
        f"{name:<36} "
        f"{fmt(av):>15} "
        f"{fmt(rv):>15} "
        f"{fmt(ratio):>10}"
    )


print()
print("Wrote:")
print(f"  {comparison_path}")
print(f"  {derived_path}")
PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo "Raw counters:"
echo "  $OUT/counter_comparison.csv"

echo "Normalized metrics:"
echo "  $OUT/derived_summary.csv"

echo "Skipped counters:"
echo "  $OUT/skipped_counters.txt"