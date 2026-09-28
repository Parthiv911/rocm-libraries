#!/usr/bin/env python3
"""
Deep analyzer for ROCprofiler stochastic PC sampling of the ROCKE attention kernel.

It reads:
  *_kernel_trace.csv
  *_pc_sampling_stochastic.csv
  *_results.json

It writes all results into the SAME rocprofv3 result directory and also prints
the main report to stdout.

Default target:
  rocke_attention_dense_d128_hq32_kv8_bn64_bf16_sq4096_sk4096_
  causal_lazyrs_persist304_gfx942_b1_wpe2_kdbvpf1

Usage:
  /root/gpu-bench/venvs/rocke/bin/python analyze_rocke_pc_sampling.py \
      --dir /root/gpu-bench/profiles/rocke_pc_sampling_1000/rocm-10-0-gpu-mi300x1-192gb-devcloud-atl1

If --dir is omitted, the script searches under /root/gpu-bench/profiles for
the newest directory containing all three required files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import pandas as pd
except ImportError:
    raise SystemExit(
        "ERROR: pandas is required. Run with the ROCKE Python environment:\n"
        "  /root/gpu-bench/venvs/rocke/bin/python analyze_rocke_pc_sampling.py ..."
    )


DEFAULT_ROOT = Path("/root/gpu-bench/profiles")
DEFAULT_KERNEL = (
    "rocke_attention_dense_d128_hq32_kv8_bn64_bf16_sq4096_sk4096_"
    "causal_lazyrs_persist304_gfx942_b1_wpe2_kdbvpf1"
)

STALL_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
INST_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"


def short_stall(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace(STALL_PREFIX, "")


def short_inst_type(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace(INST_PREFIX, "")


def pct(n: float, d: float) -> float:
    return 0.0 if not d else 100.0 * n / d


def popcount_u64(x: Any) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return 0


def numeric(v: Any):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def find_pc_wrappers(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Recursively find JSON objects of the form:
      {"record": {... PC-sampling record ...}, "inst_index": ...}

    This avoids depending on the exact top-level rocprofv3 JSON schema.
    """
    if isinstance(obj, dict):
        rec = obj.get("record")
        if isinstance(rec, dict):
            if (
                "dispatch_id" in rec
                and "timestamp" in rec
                and "pc" in rec
                and ("wave_issued" in rec or "snapshot" in rec)
            ):
                yield obj
        for v in obj.values():
            yield from find_pc_wrappers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_pc_wrappers(v)


def locate_result_dir(root: Path) -> Path:
    candidates = []
    for p in root.rglob("*_pc_sampling_stochastic.csv"):
        d = p.parent
        if list(d.glob("*_kernel_trace.csv")) and list(d.glob("*_results.json")):
            candidates.append(d)
    if not candidates:
        raise SystemExit(
            f"ERROR: could not find a rocprofv3 result directory under {root}"
        )
    return max(candidates, key=lambda p: max(x.stat().st_mtime for x in p.iterdir()))


def exactly_one(pattern: str, d: Path) -> Path:
    xs = sorted(d.glob(pattern))
    if len(xs) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {pattern} in {d}, found {len(xs)}"
        )
    return xs[0]


def add_occurrence_index(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["_match_occ"] = df.groupby(cols, dropna=False).cumcount()
    return df


def df_to_text(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "(none)"
    x = df if max_rows is None else df.head(max_rows)
    return x.to_string(index=False)


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="rocprofv3 result directory containing kernel_trace, PC CSV, and results.json",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="search root used when --dir is omitted",
    )
    ap.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL,
        help="exact substring used to select target kernel dispatches",
    )
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--phase-bins", type=int, default=10)
    opt = ap.parse_args()

    outdir = opt.dir.resolve() if opt.dir else locate_result_dir(opt.root.resolve())
    kt_path = exactly_one("*_kernel_trace.csv", outdir)
    pc_path = exactly_one("*_pc_sampling_stochastic.csv", outdir)
    js_path = exactly_one("*_results.json", outdir)

    kt = pd.read_csv(kt_path)
    pc = pd.read_csv(pc_path)

    required_kt = {"Dispatch_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"}
    required_pc = {
        "Sample_Timestamp",
        "Exec_Mask",
        "Dispatch_Id",
        "Instruction",
        "Wave_Issued_Instruction",
        "Instruction_Type",
        "Stall_Reason",
        "Wave_Count",
    }
    if not required_kt.issubset(kt.columns):
        raise SystemExit(f"ERROR: kernel trace is missing columns: {required_kt - set(kt.columns)}")
    if not required_pc.issubset(pc.columns):
        raise SystemExit(f"ERROR: PC CSV is missing columns: {required_pc - set(pc.columns)}")

    target = kt[
        kt["Kernel_Name"].astype(str).str.contains(opt.kernel, na=False, regex=False)
    ].copy()

    if target.empty:
        raise SystemExit(
            "ERROR: target ROCKE kernel was not found in kernel trace.\n"
            f"Kernel substring:\n  {opt.kernel}"
        )

    target_ids = set(int(x) for x in target["Dispatch_Id"].tolist())
    r_csv = pc[pc["Dispatch_Id"].isin(target_ids)].copy()

    if r_csv.empty:
        raise SystemExit("ERROR: no PC samples belong to selected ROCKE dispatches")

    # Normalize raw CSV fields.
    r_csv["Dispatch_Id"] = r_csv["Dispatch_Id"].astype("int64")
    r_csv["Sample_Timestamp"] = r_csv["Sample_Timestamp"].astype("int64")
    r_csv["Exec_Mask"] = r_csv["Exec_Mask"].astype("uint64")
    r_csv["Wave_Issued_Instruction"] = r_csv["Wave_Issued_Instruction"].astype("int64")
    r_csv["Wave_Count"] = pd.to_numeric(r_csv["Wave_Count"], errors="coerce")
    r_csv["stall_short"] = r_csv["Stall_Reason"].map(short_stall)
    r_csv["inst_type_short"] = r_csv["Instruction_Type"].map(short_inst_type)
    r_csv["active_lanes"] = r_csv["Exec_Mask"].map(popcount_u64)

    # Load comprehensive JSON and recursively extract stochastic PC records.
    with open(js_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_rows: List[Dict[str, Any]] = []
    for wrapper in find_pc_wrappers(data):
        rec = wrapper["record"]
        try:
            did = int(rec.get("dispatch_id"))
        except Exception:
            continue
        if did not in target_ids:
            continue

        row: Dict[str, Any] = {
            "Dispatch_Id": did,
            "Sample_Timestamp": int(rec.get("timestamp", 0)),
            "Exec_Mask": int(rec.get("exec_mask", 0)),
            "inst_index": wrapper.get("inst_index"),
            "wave_in_grp": rec.get("wave_in_grp"),
            "wave_issued_json": rec.get("wave_issued"),
            "inst_type_json": rec.get("inst_type"),
            "wave_count_json": rec.get("wave_count"),
        }

        pcj = rec.get("pc", {}) or {}
        row["code_object_id"] = pcj.get("code_object_id")
        row["code_object_offset"] = pcj.get("code_object_offset")

        wg = rec.get("wrkgrp_id", {}) or {}
        row["wg_x"] = wg.get("x")
        row["wg_y"] = wg.get("y")
        row["wg_z"] = wg.get("z")

        hw = rec.get("hw_id", {}) or {}
        for k, v in hw.items():
            row[f"hw_{k}"] = v

        flags = rec.get("flags", {}) or {}
        for k, v in flags.items():
            row[f"flag_{k}"] = v

        snap = rec.get("snapshot", {}) or {}
        for k, v in flatten_dict(snap).items():
            row[f"snap_{k}"] = v

        mem = rec.get("memory_counters", {}) or {}
        for k, v in flatten_dict(mem).items():
            row[f"mem_{k}"] = v

        json_rows.append(row)

    jdf = pd.DataFrame(json_rows)

    # Join JSON records to CSV instructions. Duplicated samples can share the same
    # timestamp/dispatch/mask, so use an occurrence index as a deterministic tiebreaker.
    match_cols = ["Dispatch_Id", "Sample_Timestamp", "Exec_Mask"]
    c = add_occurrence_index(r_csv, match_cols)

    if not jdf.empty:
        jdf["Dispatch_Id"] = pd.to_numeric(jdf["Dispatch_Id"], errors="coerce").astype("Int64")
        jdf["Sample_Timestamp"] = pd.to_numeric(jdf["Sample_Timestamp"], errors="coerce").astype("Int64")
        jdf["Exec_Mask"] = pd.to_numeric(jdf["Exec_Mask"], errors="coerce").astype("UInt64")
        jdf = add_occurrence_index(jdf, match_cols)
        enriched = c.merge(
            jdf,
            on=match_cols + ["_match_occ"],
            how="left",
            validate="one_to_one",
        )
    else:
        enriched = c.copy()

    # Add per-dispatch normalized phase.
    bounds = target[["Dispatch_Id", "Start_Timestamp", "End_Timestamp"]].copy()
    bounds["Dispatch_Id"] = bounds["Dispatch_Id"].astype("int64")
    enriched = enriched.merge(bounds, on="Dispatch_Id", how="left")
    dur = enriched["End_Timestamp"] - enriched["Start_Timestamp"]
    enriched["phase"] = (
        (enriched["Sample_Timestamp"] - enriched["Start_Timestamp"]) / dur
    )
    enriched.loc[(dur <= 0) | enriched["phase"].isna(), "phase"] = float("nan")
    enriched["phase"] = enriched["phase"].clip(lower=0.0, upper=0.999999999)

    # Output raw enriched sample table.
    enriched_path = outdir / "pc_enriched_samples.csv"
    write_csv(enriched, enriched_path)

    total = len(enriched)
    stalled = enriched[enriched["Wave_Issued_Instruction"] == 0].copy()
    issued = enriched[enriched["Wave_Issued_Instruction"] == 1].copy()

    report: List[str] = []

    def section(title: str):
        report.append("")
        report.append("=" * 88)
        report.append(title)
        report.append("=" * 88)

    report.append("ROCKE STOCHASTIC PC-SAMPLING DEEP ANALYSIS")
    report.append(f"Directory:    {outdir}")
    report.append(f"Kernel trace: {kt_path.name}")
    report.append(f"PC CSV:       {pc_path.name}")
    report.append(f"JSON:         {js_path.name}")
    report.append(f"Kernel:       {opt.kernel}")
    report.append(f"Dispatches:   {len(target)}")
    report.append(f"Raw samples:  {total}")
    report.append(
        "NOTE: all stall percentages below use one vote per PC sample. "
        "Wave_Count is occupancy context, NOT a statistical weight."
    )

    # 1. Issued vs stalled.
    section("1. ISSUED VS NOT ISSUED — RAW SAMPLE COUNTS")
    issue_counts = enriched["Wave_Issued_Instruction"].value_counts().sort_index()
    issue_rows = []
    for state, n in issue_counts.items():
        issue_rows.append(
            {
                "wave_issued": int(state),
                "samples": int(n),
                "percent": round(pct(n, total), 3),
            }
        )
    issue_df = pd.DataFrame(issue_rows)
    write_csv(issue_df, outdir / "pc_issued_vs_stalled.csv")
    report.append(df_to_text(issue_df))

    # 2. Stall reasons.
    section("2. STALL REASONS — RAW SAMPLE COUNTS")
    sr = stalled["stall_short"].value_counts()
    stall_reason_df = pd.DataFrame(
        {
            "stall_reason": sr.index,
            "samples": sr.values.astype(int),
            "percent_of_stalled": [round(pct(v, len(stalled)), 3) for v in sr.values],
            "percent_of_all_samples": [round(pct(v, total), 3) for v in sr.values],
        }
    )
    write_csv(stall_reason_df, outdir / "pc_stall_reasons.csv")
    report.append(df_to_text(stall_reason_df))

    # 3. Instruction types among issued samples.
    section("3. ISSUED INSTRUCTION TYPES")
    it = issued["inst_type_short"].value_counts()
    inst_type_df = pd.DataFrame(
        {
            "instruction_type": it.index,
            "samples": it.values.astype(int),
            "percent_of_issued": [round(pct(v, len(issued)), 3) for v in it.values],
        }
    )
    write_csv(inst_type_df, outdir / "pc_issued_instruction_types.csv")
    report.append(df_to_text(inst_type_df))

    # 4. Top stalled instructions.
    section(f"4. TOP {opt.top} STALLED INSTRUCTION + REASON PAIRS")
    top_pair = (
        stalled.groupby(["Instruction", "stall_short"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values("samples", ascending=False)
    )
    top_pair["percent_of_stalled"] = (100.0 * top_pair["samples"] / len(stalled)).round(3)
    write_csv(top_pair, outdir / "pc_stall_by_instruction_reason.csv")
    report.append(df_to_text(top_pair, opt.top))

    # 5. Top exact PCs — this distinguishes repeated identical wait/barrier instructions.
    section(f"5. TOP {opt.top} EXACT PCs")
    have_pc = "code_object_offset" in enriched.columns and enriched["code_object_offset"].notna().any()
    if have_pc:
        epc = enriched.copy()
        epc["code_object_id"] = pd.to_numeric(epc["code_object_id"], errors="coerce")
        epc["code_object_offset"] = pd.to_numeric(epc["code_object_offset"], errors="coerce")
        key = ["code_object_id", "code_object_offset", "Instruction"]
        exact = (
            epc.groupby(key, dropna=False)
            .agg(
                samples=("Dispatch_Id", "size"),
                stalled_samples=("Wave_Issued_Instruction", lambda s: int((s == 0).sum())),
                issued_samples=("Wave_Issued_Instruction", lambda s: int((s == 1).sum())),
                mean_wave_count=("Wave_Count", "mean"),
                mean_active_lanes=("active_lanes", "mean"),
            )
            .reset_index()
        )
        exact["stall_percent"] = (
            100.0 * exact["stalled_samples"] / exact["samples"]
        ).round(3)

        # Dominant stall reason at each PC.
        by_reason = (
            epc[epc["Wave_Issued_Instruction"] == 0]
            .groupby(key + ["stall_short"], dropna=False)
            .size()
            .reset_index(name="reason_samples")
            .sort_values("reason_samples", ascending=False)
        )
        if not by_reason.empty:
            dom = by_reason.drop_duplicates(key).rename(
                columns={"stall_short": "dominant_stall_reason"}
            )
            exact = exact.merge(
                dom[key + ["dominant_stall_reason", "reason_samples"]],
                on=key,
                how="left",
            )
        exact = exact.sort_values(
            ["stalled_samples", "samples"], ascending=[False, False]
        )
        write_csv(exact, outdir / "pc_exact_pc.csv")
        report.append(df_to_text(exact, opt.top))
    else:
        report.append("JSON did not expose usable code_object_offset values.")

    # 6. Occupancy context (Wave_Count).
    section("6. OCCUPANCY CONTEXT — Wave_Count")
    occ = (
        enriched.assign(state=enriched["Wave_Issued_Instruction"].map({0: "stalled", 1: "issued"}))
        .groupby("state")["Wave_Count"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    write_csv(occ, outdir / "pc_occupancy_issued_stalled.csv")
    report.append(df_to_text(occ))

    occ_reason = (
        stalled.groupby("stall_short")["Wave_Count"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    write_csv(occ_reason, outdir / "pc_occupancy_by_stall_reason.csv")
    report.append("\nOccupancy by stall reason:")
    report.append(df_to_text(occ_reason))

    # 7. EXEC-mask lane utilization.
    section("7. EXEC MASK / ACTIVE-LANE UTILIZATION")
    lanes = (
        enriched["active_lanes"]
        .value_counts()
        .sort_index()
        .rename_axis("active_lanes")
        .reset_index(name="samples")
    )
    lanes["percent"] = (100.0 * lanes["samples"] / total).round(3)
    write_csv(lanes, outdir / "pc_exec_mask_lanes.csv")

    report.append(
        f"Mean active lanes:   {enriched['active_lanes'].mean():.3f} / 64"
    )
    report.append(
        f"Median active lanes: {enriched['active_lanes'].median():.3f} / 64"
    )
    report.append(
        f"Full-wave samples:   {(enriched['active_lanes'] == 64).sum()} "
        f"({pct((enriched['active_lanes'] == 64).sum(), total):.3f}%)"
    )
    report.append("\nActive-lane distribution:")
    report.append(df_to_text(lanes))

    lane_reason = (
        stalled.groupby("stall_short")["active_lanes"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    write_csv(lane_reason, outdir / "pc_active_lanes_by_stall_reason.csv")
    report.append("\nActive lanes by stall reason:")
    report.append(df_to_text(lane_reason))

    # 8. Wave position within workgroup.
    section("8. WAVE POSITION WITHIN WORKGROUP")
    if "wave_in_grp" in enriched.columns and enriched["wave_in_grp"].notna().any():
        wrows = []
        for wave, g in enriched.dropna(subset=["wave_in_grp"]).groupby("wave_in_grp"):
            n = len(g)
            gs = g[g["Wave_Issued_Instruction"] == 0]
            wrows.append(
                {
                    "wave_in_grp": int(wave),
                    "samples": n,
                    "stall_percent": round(pct(len(gs), n), 3),
                    "barrier_percent_all": round(
                        pct((gs["stall_short"] == "BARRIER_WAIT").sum(), n), 3
                    ),
                    "waitcnt_percent_all": round(
                        pct((gs["stall_short"] == "WAITCNT").sum(), n), 3
                    ),
                    "arbiter_not_win_percent_all": round(
                        pct((gs["stall_short"] == "ARBITER_NOT_WIN").sum(), n), 3
                    ),
                    "mean_wave_count": round(g["Wave_Count"].mean(), 3),
                    "mean_active_lanes": round(g["active_lanes"].mean(), 3),
                }
            )
        wave_df = pd.DataFrame(wrows).sort_values("wave_in_grp")
        write_csv(wave_df, outdir / "pc_wave_position.csv")
        report.append(df_to_text(wave_df))
    else:
        report.append("wave_in_grp is unavailable in the JSON output.")

    # 9. Arbiter pipeline issue/backpressure bits.
    section("9. ARBITER PIPELINE ISSUE / BACKPRESSURE")
    issue_cols = [c for c in enriched.columns if c.startswith("snap_arb_state_issue_")]
    stall_cols = [c for c in enriched.columns if c.startswith("snap_arb_state_stall_")]

    arb_rows = []
    suffixes = sorted(
        set(c.removeprefix("snap_arb_state_issue_") for c in issue_cols)
        | set(c.removeprefix("snap_arb_state_stall_") for c in stall_cols)
    )
    for suffix in suffixes:
        ic = f"snap_arb_state_issue_{suffix}"
        sc = f"snap_arb_state_stall_{suffix}"
        i = pd.to_numeric(enriched[ic], errors="coerce").fillna(0) if ic in enriched else pd.Series(0, index=enriched.index)
        s = pd.to_numeric(enriched[sc], errors="coerce").fillna(0) if sc in enriched else pd.Series(0, index=enriched.index)
        arb_rows.append(
            {
                "pipeline": suffix,
                "issue_bit_samples": int((i != 0).sum()),
                "issue_bit_percent": round(pct((i != 0).sum(), total), 3),
                "backpressure_bit_samples": int((s != 0).sum()),
                "backpressure_bit_percent": round(pct((s != 0).sum(), total), 3),
            }
        )

    arb_df = pd.DataFrame(arb_rows)
    if not arb_df.empty:
        arb_df = arb_df.sort_values(
            ["backpressure_bit_samples", "issue_bit_samples"],
            ascending=False,
        )
        write_csv(arb_df, outdir / "pc_arbiter_pipelines.csv")
        report.append(df_to_text(arb_df))
    else:
        report.append(
            "No arb_state_issue_* / arb_state_stall_* fields were found in JSON."
        )

    # 10. Memory counters.
    section("10. OUTSTANDING MEMORY COUNTERS")
    mem_cols = [c for c in enriched.columns if c.startswith("mem_")]
    has_mem_flag = None
    for c in enriched.columns:
        if c == "flag_has_mem_cnt" or c.endswith(".has_mem_cnt"):
            has_mem_flag = c
            break

    mem_meaningful = True
    if has_mem_flag is not None:
        vals = pd.to_numeric(enriched[has_mem_flag], errors="coerce").fillna(0)
        mem_meaningful = bool((vals != 0).any())

    if mem_cols and mem_meaningful:
        mrows = []
        for c in mem_cols:
            v = pd.to_numeric(enriched[c], errors="coerce")
            if v.notna().any():
                mrows.append(
                    {
                        "counter": c.removeprefix("mem_"),
                        "samples_present": int(v.notna().sum()),
                        "mean": round(v.mean(), 3),
                        "median": round(v.median(), 3),
                        "p95": round(v.quantile(0.95), 3),
                        "max": v.max(),
                    }
                )
        mem_df = pd.DataFrame(mrows).sort_values("mean", ascending=False)
        write_csv(mem_df, outdir / "pc_memory_counters.csv")
        report.append(df_to_text(mem_df))

        # Counter means by stall reason.
        rows = []
        for reason, g in stalled.groupby("stall_short"):
            row = {"stall_reason": reason, "samples": len(g)}
            for c in mem_cols:
                v = pd.to_numeric(g[c], errors="coerce")
                if v.notna().any():
                    row[f"mean_{c.removeprefix('mem_')}"] = round(v.mean(), 3)
            rows.append(row)
        mem_reason_df = pd.DataFrame(rows).sort_values("samples", ascending=False)
        write_csv(mem_reason_df, outdir / "pc_memory_counters_by_stall_reason.csv")
        report.append("\nMemory counters by stall reason:")
        report.append(df_to_text(mem_reason_df))
    else:
        why = (
            "flags.has_mem_cnt is zero for this collection"
            if has_mem_flag is not None
            else "memory counter fields are absent"
        )
        report.append(f"Memory counters unavailable: {why}.")

    # 11. Hardware locality — chiplet/XCD, SE, SA, CU/WGP, SIMD.
    section("11. HARDWARE LOCALITY")
    locality_fields = [
        "hw_chiplet",
        "hw_shader_engine_id",
        "hw_shader_array_id",
        "hw_cu_or_wgp_id",
        "hw_simd_id",
    ]
    available_locality = [c for c in locality_fields if c in enriched.columns]
    if available_locality:
        # Chiplet summary.
        if "hw_chiplet" in enriched.columns:
            rows = []
            for chiplet, g in enriched.dropna(subset=["hw_chiplet"]).groupby("hw_chiplet"):
                gs = g[g["Wave_Issued_Instruction"] == 0]
                rows.append(
                    {
                        "chiplet": int(chiplet),
                        "samples": len(g),
                        "sample_percent": round(pct(len(g), total), 3),
                        "stall_percent": round(pct(len(gs), len(g)), 3),
                        "mean_wave_count": round(g["Wave_Count"].mean(), 3),
                    }
                )
            chip_df = pd.DataFrame(rows).sort_values("chiplet")
            write_csv(chip_df, outdir / "pc_chiplet.csv")
            report.append("Chiplet/XCD:")
            report.append(df_to_text(chip_df))

        # CU/WGP composite summary.
        cu_keys = [
            c
            for c in ["hw_chiplet", "hw_shader_engine_id", "hw_shader_array_id", "hw_cu_or_wgp_id"]
            if c in enriched.columns
        ]
        if "hw_cu_or_wgp_id" in cu_keys:
            rows = []
            for key, g in enriched.dropna(subset=cu_keys).groupby(cu_keys):
                if not isinstance(key, tuple):
                    key = (key,)
                gs = g[g["Wave_Issued_Instruction"] == 0]
                row = {k: int(v) for k, v in zip(cu_keys, key)}
                row.update(
                    {
                        "samples": len(g),
                        "stall_percent": round(pct(len(gs), len(g)), 3),
                        "mean_wave_count": round(g["Wave_Count"].mean(), 3),
                    }
                )
                rows.append(row)
            cu_df = pd.DataFrame(rows).sort_values("samples", ascending=False)
            write_csv(cu_df, outdir / "pc_cu_wgp.csv")
            report.append("\nTop CU/WGP locations by sample count:")
            report.append(df_to_text(cu_df, min(opt.top, 30)))
    else:
        report.append("Hardware locality fields were not found in JSON.")

    # 12. Time-normalized phase of each dispatch.
    section("12. NORMALIZED KERNEL PHASE")
    phase_valid = enriched.dropna(subset=["phase"]).copy()
    if not phase_valid.empty:
        nb = max(1, opt.phase_bins)
        phase_valid["phase_bin"] = (phase_valid["phase"] * nb).astype(int).clip(0, nb - 1)
        rows = []
        for b, g in phase_valid.groupby("phase_bin"):
            gs = g[g["Wave_Issued_Instruction"] == 0]
            row = {
                "phase_bin": int(b),
                "phase_start_pct": round(100.0 * b / nb, 1),
                "phase_end_pct": round(100.0 * (b + 1) / nb, 1),
                "samples": len(g),
                "sample_percent": round(pct(len(g), len(phase_valid)), 3),
                "stall_percent": round(pct(len(gs), len(g)), 3),
                "barrier_percent_all": round(
                    pct((gs["stall_short"] == "BARRIER_WAIT").sum(), len(g)), 3
                ),
                "waitcnt_percent_all": round(
                    pct((gs["stall_short"] == "WAITCNT").sum(), len(g)), 3
                ),
                "arbiter_not_win_percent_all": round(
                    pct((gs["stall_short"] == "ARBITER_NOT_WIN").sum(), len(g)), 3
                ),
                "arbiter_win_ex_stall_percent_all": round(
                    pct((gs["stall_short"] == "ARBITER_WIN_EX_STALL").sum(), len(g)), 3
                ),
                "mean_wave_count": round(g["Wave_Count"].mean(), 3),
                "mean_active_lanes": round(g["active_lanes"].mean(), 3),
            }
            rows.append(row)
        phase_df = pd.DataFrame(rows).sort_values("phase_bin")
        write_csv(phase_df, outdir / "pc_phase_bins.csv")
        report.append(df_to_text(phase_df))
    else:
        report.append("Could not match sample timestamps to kernel start/end timestamps.")

    # 13. Barrier / waitcnt hotspots by exact PC.
    section("13. BARRIER + WAITCNT HOTSPOTS BY EXACT PC")
    if have_pc:
        bw = enriched[
            (enriched["Wave_Issued_Instruction"] == 0)
            & (enriched["stall_short"].isin(["BARRIER_WAIT", "WAITCNT"]))
        ].copy()
        if not bw.empty:
            bw_hot = (
                bw.groupby(
                    ["code_object_id", "code_object_offset", "Instruction", "stall_short"],
                    dropna=False,
                )
                .agg(
                    samples=("Dispatch_Id", "size"),
                    mean_wave_count=("Wave_Count", "mean"),
                    mean_active_lanes=("active_lanes", "mean"),
                )
                .reset_index()
                .sort_values("samples", ascending=False)
            )
            bw_hot["percent_of_all_stalled"] = (
                100.0 * bw_hot["samples"] / len(stalled)
            ).round(3)
            write_csv(bw_hot, outdir / "pc_barrier_waitcnt_exact_pc.csv")
            report.append(df_to_text(bw_hot, opt.top))
        else:
            report.append("No BARRIER_WAIT or WAITCNT samples.")
    else:
        report.append("Exact-PC data unavailable.")

    # 14. Join quality / diagnostics.
    section("14. JSON JOIN QUALITY / DIAGNOSTICS")
    if "code_object_offset" in enriched.columns:
        joined = int(enriched["code_object_offset"].notna().sum())
        report.append(
            f"Samples matched to JSON detail: {joined}/{total} ({pct(joined,total):.3f}%)"
        )
    else:
        report.append("No JSON PC detail columns were merged.")

    if "flag_has_mem_cnt" in enriched.columns:
        f = pd.to_numeric(enriched["flag_has_mem_cnt"], errors="coerce").fillna(0)
        report.append(
            f"Samples with has_mem_cnt != 0: {(f != 0).sum()}/{total}"
        )

    report.append("")
    report.append("GENERATED FILES")
    report.append("- pc_sampling_analysis.txt")
    report.append("- pc_enriched_samples.csv")
    report.append("- pc_issued_vs_stalled.csv")
    report.append("- pc_stall_reasons.csv")
    report.append("- pc_issued_instruction_types.csv")
    report.append("- pc_stall_by_instruction_reason.csv")
    report.append("- pc_exact_pc.csv (if JSON PC offsets available)")
    report.append("- pc_occupancy_issued_stalled.csv")
    report.append("- pc_occupancy_by_stall_reason.csv")
    report.append("- pc_exec_mask_lanes.csv")
    report.append("- pc_active_lanes_by_stall_reason.csv")
    report.append("- pc_wave_position.csv (if JSON field available)")
    report.append("- pc_arbiter_pipelines.csv (if snapshot bits available)")
    report.append("- pc_memory_counters*.csv (if has_mem_cnt is supported)")
    report.append("- pc_chiplet.csv / pc_cu_wgp.csv (if hardware IDs available)")
    report.append("- pc_phase_bins.csv")
    report.append("- pc_barrier_waitcnt_exact_pc.csv")

    text = "\n".join(report) + "\n"
    report_path = outdir / "pc_sampling_analysis.txt"
    report_path.write_text(text, encoding="utf-8")

    print(text, end="")
    print(f"\nSaved full report to:\n  {report_path}")
    print(f"Saved enriched per-sample table to:\n  {enriched_path}")


if __name__ == "__main__":
    main()