#!/usr/bin/env python3
"""
Analyze ROCprofv3 stochastic PC-sampling output for GPU stall behavior.

Reports:
  1. Overall sampled-wave stall reasons
  2. Issued instruction-type distribution
  3. WAITCNT breakdown by exact wait instruction
  4. WAITCNT breakdown grouped into VMEM / LGKM / other
  5. Missing/undecoded WAITCNT samples
  6. Hardware arbiter stall-state snapshot frequencies from JSON
  7. ARBITER_WIN_EX_STALL correlation with hardware pipe stall bits
  8. Exact instructions seen during ARBITER_WIN_EX_STALL samples

Typical usage:

  python analyze_pc_sampling.py \
    --csv /root/gpu-bench/profiles/rocke_pc_sampling/.../11478_pc_sampling_stochastic.csv \
    --json /root/gpu-bench/profiles/rocke_pc_sampling/.../11478_results.json

Or point it at the generated profile directory:

  python analyze_pc_sampling.py \
    --profile-dir /root/gpu-bench/profiles/rocke_pc_sampling/rocm-10-0-gpu-mi300x1-192gb-devcloud-atl1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ARB_FIELDS = [
    "arb_state_stall_flat",
    "arb_state_stall_vmem_tex",
    "arb_state_stall_lds",
    "arb_state_stall_lds_direct",
    "arb_state_stall_matrix",
    "arb_state_stall_valu",
    "arb_state_stall_scalar",
    "arb_state_stall_exp",
    "arb_state_stall_misc",
    "arb_state_stall_brmsg",
]


def find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {directory}")

    # Prefer newest file if multiple profiling runs exist.
    return max(matches, key=lambda p: p.stat().st_mtime)


def short_reason(x: Any) -> str:
    if pd.isna(x):
        return "<missing>"
    s = str(x)
    prefix = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
    return s.removeprefix(prefix)


def short_type(x: Any) -> str:
    if pd.isna(x):
        return "<missing>"
    s = str(x)
    prefix = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"
    return s.removeprefix(prefix)


def print_series_counts(title: str, series: pd.Series, total: int | None = None) -> None:
    print(f"\n=== {title} ===")
    if series.empty:
        print("(none)")
        return

    if total is None:
        total = int(series.sum())

    width = max(len(str(x)) for x in series.index)
    for key, value in series.items():
        pct = (100.0 * value / total) if total else 0.0
        print(f"{str(key):<{width}}  {int(value):6d}  {pct:6.2f}%")


def analyze_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {
        "Wave_Issued_Instruction",
        "Instruction_Type",
        "Stall_Reason",
        "Instruction",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"CSV missing required columns: {sorted(missing)}\n"
            f"Columns present: {df.columns.tolist()}"
        )

    print("============================================================")
    print("PC-SAMPLING CSV")
    print("============================================================")
    print(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    issued_mask = df["Wave_Issued_Instruction"] == 1
    stalled_mask = df["Wave_Issued_Instruction"] == 0

    issued = df[issued_mask].copy()
    stalled = df[stalled_mask].copy()

    print(f"\nIssued samples:  {len(issued)}")
    print(f"Stalled samples: {len(stalled)}")

    # 1. Overall stall reasons.
    reasons = stalled["Stall_Reason"].map(short_reason).value_counts(dropna=False)
    print_series_counts("STALL REASONS", reasons, len(stalled))

    # 2. Issued instruction types.
    issued_types = issued["Instruction_Type"].map(short_type).value_counts(dropna=False)
    print_series_counts("ISSUED INSTRUCTION TYPES", issued_types, len(issued))

    # 3. WAITCNT analysis.
    wait = stalled[
        stalled["Stall_Reason"].astype(str).str.contains("WAITCNT", na=False)
    ].copy()

    print("\n============================================================")
    print("WAITCNT ANALYSIS")
    print("============================================================")
    print(f"Total WAITCNT samples: {len(wait)}")

    wait_instr = wait["Instruction"].value_counts(dropna=False)
    print_series_counts("WAITCNT BY EXACT INSTRUCTION", wait_instr, len(wait))

    def wait_group(instr: Any) -> str:
        if pd.isna(instr):
            return "MISSING/UNDECODED"
        s = str(instr)
        if "vmcnt" in s:
            return "VMEM"
        if "lgkmcnt" in s:
            return "LGKM"
        return "OTHER"

    wait_groups = wait["Instruction"].map(wait_group).value_counts()
    print_series_counts("WAITCNT GROUPED", wait_groups, len(wait))

    missing_wait = wait[wait["Instruction"].isna()]
    print(f"\nMissing/undecoded WAITCNT samples: {len(missing_wait)}")
    if not missing_wait.empty:
        cols = [
            c
            for c in [
                "Instruction",
                "Instruction_Comment",
                "Instruction_Type",
                "Wave_Count",
                "Sample_Timestamp",
                "Dispatch_Id",
            ]
            if c in missing_wait.columns
        ]
        print("\nFirst 30 missing/undecoded WAITCNT rows:")
        print(missing_wait[cols].head(30).to_string(index=False))

    # 4. ARBITER_WIN_EX_STALL from CSV.
    ex_stall = stalled[
        stalled["Stall_Reason"].astype(str).str.contains(
            "ARBITER_WIN_EX_STALL", na=False
        )
    ].copy()

    print("\n============================================================")
    print("ARBITER_WIN_EX_STALL — CSV VIEW")
    print("============================================================")
    print(f"Samples: {len(ex_stall)}")

    if not ex_stall.empty:
        ex_types = (
            ex_stall["Instruction_Type"]
            .map(short_type)
            .value_counts(dropna=False)
        )
        print_series_counts(
            "ARBITER_WIN_EX_STALL INSTRUCTION TYPES",
            ex_types,
            len(ex_stall),
        )

        ex_instr = ex_stall["Instruction"].value_counts(dropna=False)
        print_series_counts(
            "ARBITER_WIN_EX_STALL EXACT INSTRUCTIONS",
            ex_instr,
            len(ex_stall),
        )

    return df


def walk_dicts(x: Any):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk_dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk_dicts(v)


def analyze_json(json_path: Path) -> None:
    with json_path.open() as fh:
        obj = json.load(fh)

    print("\n============================================================")
    print("PC-SAMPLING JSON / HARDWARE ARBITER STATE")
    print("============================================================")
    print(json_path)

    snapshots = [
        d for d in walk_dicts(obj)
        if any(k in d for k in ARB_FIELDS)
    ]

    print(f"Hardware snapshots: {len(snapshots)}")

    counts = {}
    for field in ARB_FIELDS:
        counts[field] = sum(bool(d.get(field, 0)) for d in snapshots)

    print("\n=== HARDWARE PIPE STALL STATES ===")
    for field, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * count / len(snapshots) if snapshots else 0.0
        print(f"{field:30s} {count:6d}  {pct:6.2f}%")

    # Find records that contain both stall reason and arbiter state.
    records = [
        d for d in snapshots
        if "stall_reason" in d
    ]

    target = [
        d for d in records
        if "ARBITER_WIN_EX_STALL" in str(d.get("stall_reason", ""))
    ]

    print("\n=== ARBITER_WIN_EX_STALL -> PIPE ===")
    print(f"ARBITER_WIN_EX_STALL snapshots: {len(target)}")

    for field in ARB_FIELDS:
        count = sum(bool(d.get(field, 0)) for d in target)
        pct = 100.0 * count / len(target) if target else 0.0
        print(f"{field:30s} {count:6d}  {pct:6.2f}%")

    if target:
        memory_fields = {
            "FLAT": "arb_state_stall_flat",
            "VMEM/TEX": "arb_state_stall_vmem_tex",
            "LDS": "arb_state_stall_lds",
            "LDS_DIRECT": "arb_state_stall_lds_direct",
            "MATRIX": "arb_state_stall_matrix",
            "VALU": "arb_state_stall_valu",
            "SCALAR": "arb_state_stall_scalar",
        }

        print("\n=== COMPACT PIPE-BACKPRESSURE SUMMARY ===")
        for label, field in memory_fields.items():
            count = sum(bool(d.get(field, 0)) for d in target)
            pct = 100.0 * count / len(target)
            print(f"{label:12s} {count:6d}  {pct:6.2f}%")

        flat_tex = sum(
            1
            for d in target
            if bool(d.get("arb_state_stall_flat", 0))
            or bool(d.get("arb_state_stall_vmem_tex", 0))
        )
        print(
            f"\nSamples with FLAT and/or VMEM/TEX backpressure: "
            f"{flat_tex}/{len(target)} = {100.0 * flat_tex / len(target):.2f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze ROCprofv3 stochastic PC-sampling CSV/JSON output."
    )
    ap.add_argument("--csv", type=Path, help="*_pc_sampling_stochastic.csv")
    ap.add_argument("--json", type=Path, help="*_results.json")
    ap.add_argument(
        "--profile-dir",
        type=Path,
        help="Directory containing rocprofv3 PC-sampling CSV and JSON",
    )
    args = ap.parse_args()

    csv_path = args.csv
    json_path = args.json

    if args.profile_dir:
        d = args.profile_dir
        if csv_path is None:
            csv_path = find_single(d, "*_pc_sampling_stochastic.csv")
        if json_path is None:
            json_path = find_single(d, "*_results.json")

    if csv_path is None and json_path is None:
        ap.error("Provide --profile-dir, --csv, --json, or both --csv and --json.")

    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        analyze_csv(csv_path)

    if json_path is not None:
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        analyze_json(json_path)


if __name__ == "__main__":
    main()