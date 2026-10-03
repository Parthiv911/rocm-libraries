#!/usr/bin/env bash
set -Eeuo pipefail

source /root/gpu-bench/env/rocke_env.sh

SWEEP="/root/gpu-bench/rocm-libraries/dnn-providers/hip-kernel-provider/rocke/library/builders/gfx942/attention/prefill/sweep_attention_dense_levers.py"
TMP="/tmp/gfx942_attention_dense_lever_sweep"
OUTROOT="/root/gpu-bench/step0_gfx942_bf16_d128"

mkdir -p "$OUTROOT"

# Representative BF16 D128 cohort already used for optimization validation.
# Format: B S HQ HKV
SHAPES=(
  "1 4096 32 8"
  "1 4096 32 16"
  "1 8192 32 8"
  "1 8192 32 16"
  "1 16384 32 8"
  "16 4096 32 8"
  "16 4096 32 16"
  "16 8192 32 8"
)

for shape in "${SHAPES[@]}"; do
  read -r B S HQ HKV <<< "$shape"

  TAG="b${B}_s${S}_hq${HQ}_hkv${HKV}_d128"
  DEST="$OUTROOT/$TAG"

  echo
  echo "======================================================================"
  echo "STEP-0 FULL SWEEP: $TAG"
  echo "======================================================================"

  rm -rf "$TMP"

  python3 "$SWEEP" \
    --batch "$B" \
    --s "$S" \
    --hq "$HQ" \
    --hkv "$HKV" \
    --d 128 \
    --dtype bf16 \
    --strategy full \
    --screen-warmup 2 \
    --screen-iters 10 \
    --top 10 \
    --rounds 5 \
    --round-warmup 10 \
    --round-iters 50

  rm -rf "$DEST"
  mkdir -p "$DEST"
  cp -a "$TMP"/. "$DEST"/

  echo "saved: $DEST"
done

python3 - "$OUTROOT" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []

for p in sorted(root.glob("*/final_ab.csv")):
    with p.open(newline="") as f:
        r = next(csv.DictReader(f), None)
    if not r:
        continue

    tag = p.parent.name
    t = tag.split("_")
    vals = {
        "B": t[0][1:],
        "S": t[1][1:],
        "HQ": t[2][2:],
        "HKV": t[3][3:],
        "D": t[4][1:],
    }

    rows.append({
        **vals,
        "median_ratio": r["median_ratio"],
        "median_candidate_ms": r["median_candidate_ms"],
        "median_baseline_ms": r["median_baseline_ms"],
        "block_m": r["block_m"],
        "block_n": r["block_n"],
        "waves_per_eu": r["waves_per_eu"],
        "lds_row_pad": r["lds_row_pad"],
        "use_cfvst": r["use_cfvst"],
        "use_v_swizzle": r["use_v_swizzle"],
        "use_exp2_fast": r["use_exp2_fast"],
        "iglp": r["iglp"],
        "persistent": r["persistent"],
        "num_persistent": r["num_persistent"],
        "persist_decode": r["persist_decode"],
        "interleave": r["interleave"],
        "result_dir": str(p.parent),
    })

out = root / "cohort_summary.csv"
if rows:
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

print()
print("======================================================================")
print("COHORT SUMMARY")
print("======================================================================")
for r in rows:
    print(
        f"B={r['B']:>2} S={r['S']:>5} HQ={r['HQ']:>2} HKV={r['HKV']:>2} "
        f"best={float(r['median_ratio']):.4f}x "
        f"bm={r['block_m']} bn={r['block_n']} "
        f"wpe={r['waves_per_eu']} pad={r['lds_row_pad']} "
        f"cfvst={r['use_cfvst']} exp2={r['use_exp2_fast']} iglp={r['iglp']}"
    )

print(f"\nsummary: {out}")
PY

echo
echo "All local Step-0 artifacts:"
echo "  $OUTROOT"