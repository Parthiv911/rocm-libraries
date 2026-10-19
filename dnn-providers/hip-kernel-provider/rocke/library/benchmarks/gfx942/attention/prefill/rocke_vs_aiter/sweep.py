#!/usr/bin/env python3
"""Staged gfx942 BF16 D128 causal prefill baseline lever sweep (12 knobs).

Twelve explicit measured shapes: B in (1,16), S in (1024,2048,4096,8192,16384),
Hq=32, Hkv in (8,16) where listed in the benchmark cohort.
The default run sweeps all 12, not the Cartesian product of these dimensions.
Base spec is production dispatch-resolved gfx942_dense for each shape.  The
search NEVER overrides CFVST, K-LDS buffer count, V-register prefetch, V
swizzle, V-row padding, or persistent-grid scheduling.  It requires the
resolved BF16-D128 base to use the non-CFVST single-K-buffer path.

Focused geometry: block_m=(64,128,256), block_n=(32,64),
lds_row_pad=(0,4,8,16), waves_per_eu=(1,2,3,4).
The broad A1 screen samples wpe=(1,2,4); A2 retests 1..4 on finalists.

Stages (beam candidates carry their *full* knob selections forward):
  A1: focused (block_m, block_n, lds_row_pad) x wpe=(1,2,4)
  A2: A1 finalists x wpe=(1..4) x pv_loop_order x causal_diag_split
   C: A2 finalists x exp2 policy choices x causal_diag_split x wpe
  B1: C finalists x scheduling (IGLP/fences) x PV order x PV priority
  B2: B1 finalists x wpe x PV order
   D: B2 finalists x output store width
   R: bounded neighborhood cross-group refinement, two rounds by default.

Every legal candidate gets a chunked FP32 SDPA correctness check and short
GPU-event screen; a geometry-diverse shortlist receives alternating paired
A/B timing. Finalists are independently revalidated and A/B timed again.
The beam/refinement strategy is BOUNDED, not a proof of a global maximum.

Place at rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter/sweep.py
Usage:
  python sweep.py --mode baseline
  python sweep.py --mode discover --dry-run
  python sweep.py --mode all
  python sweep.py --mode all --resume
Results default to sweep_results/ next to this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
import platform
import shlex
from collections import OrderedDict
from dataclasses import fields, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ARCH, H_Q, D, DTYPE = "gfx942", 32, 128, "bf16"
# Exact 12 shapes from the baseline-vs-new performance table.
# Explicit listing prevents sweeping nonexistent combinations.
COHORT_SHAPES = (
    (1, 1024, H_Q, 8, D),
    (1, 1024, H_Q, 16, D),
    (1, 2048, H_Q, 8, D),
    (1, 2048, H_Q, 16, D),
    (1, 4096, H_Q, 8, D),
    (1, 4096, H_Q, 16, D),
    (1, 8192, H_Q, 8, D),
    (1, 8192, H_Q, 16, D),
    (1, 16384, H_Q, 8, D),
    (16, 4096, H_Q, 8, D),
    (16, 4096, H_Q, 16, D),
    (16, 8192, H_Q, 8, D),
)
FENCES = (0,) + tuple(1 << i for i in range(11)) + (0x108,)
# Focused baseline sweep: broaden only if a follow-up study warrants it.
# BN=128 (pad=0 only) is intentionally out of the default search.
BLOCK_M = (64, 128, 256)
BLOCK_N = (32, 64)
LDS_ROW_PAD = (0, 4, 8, 16)
WPE = (1, 2, 3, 4)
A1_WPE = (1, 2, 4)  # screen cheaply; A2 includes the remaining value 3
PV_ORDER = ("d_major", "k_major")
DIAG_SPLIT = (False, True)
STORE_WIDTH = (1, 2, 4)
STAGE_NAMES = ("A1", "A2", "C", "B1", "B2", "D", "R1", "R2")
SWEEP_KNOBS = frozenset({
    "block_m", "block_n", "waves_per_eu", "lds_row_pad",
    "pv_loop_order", "causal_diag_split", "use_exp2_fast",
    "iglp", "iglp_mode", "pv_priority", "pv_sched_fence_mask",
    "o_store_width",
})


def product(axes):
    for values in itertools.product(*axes.values()):
        yield dict(zip(axes, values))


def scheduling():
    """16 legal IGLP/fence combinations; never turn both on."""
    yield {"iglp": False, "iglp_mode": 0, "pv_sched_fence_mask": None}
    for fence in FENCES:
        yield {"iglp": False, "iglp_mode": 0, "pv_sched_fence_mask": fence}
    for mode in (0, 1):
        yield {"iglp": True, "iglp_mode": mode, "pv_sched_fence_mask": None}


def unique(items):
    seen = set()
    for knobs in items:
        key = json.dumps(knobs, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield knobs


def stage_variants(name, seeds):
    """Produce complete overrides, preserving every incoming beam configuration."""
    if name == "A1":
        for bm, bn, pad, wpe in itertools.product(
                BLOCK_M, BLOCK_N, LDS_ROW_PAD, A1_WPE):
            yield {"block_m": bm, "block_n": bn, "lds_row_pad": pad,
                   "waves_per_eu": wpe}
        return
    for entry in seeds:
        previous = entry["knobs"]
        if name == "A2":
            overrides = product({"waves_per_eu": WPE,
                                 "pv_loop_order": PV_ORDER,
                                 "causal_diag_split": DIAG_SPLIT})
        elif name == "C":
            overrides = product({"use_exp2_fast": (None, False, True),
                                 "causal_diag_split": DIAG_SPLIT,
                                 "waves_per_eu": WPE})
        elif name == "B1":
            overrides = ({**sch, "pv_loop_order": order, "pv_priority": priority}
                         for sch, order, priority in itertools.product(
                             scheduling(), PV_ORDER, range(4)))
        elif name == "B2":
            overrides = product({"waves_per_eu": WPE,
                                 "pv_loop_order": PV_ORDER})
        elif name == "D":
            overrides = product({"o_store_width": STORE_WIDTH})
        else:
            raise ValueError(f"No generator for stage {name}")
        for override in overrides:
            yield {**previous, **override}


def refinement_variants(seeds):
    """Bounded single-axis neighbors around several *complete* finalists.

    Two passes let an A/C change be followed by a B/WPE change.  This does not
    claim pairwise-exhaustive interaction coverage.
    """
    for entry in seeds:
        old = entry["knobs"]
        yield dict(old)
        for axis, domain in (
                ("block_m", BLOCK_M), ("block_n", BLOCK_N),
                ("lds_row_pad", LDS_ROW_PAD), ("waves_per_eu", WPE),
                ("pv_loop_order", PV_ORDER), ("causal_diag_split", DIAG_SPLIT),
                ("use_exp2_fast", (None, False, True)),
                ("o_store_width", STORE_WIDTH)):
            for value in domain:
                yield {**old, axis: value}


def find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "library").is_dir() and (p / "platform/python").is_dir():
            return p
    raise RuntimeError("Cannot locate ROCKE root (library/ and platform/python/) from sweep.py location")


def load():
    root = find_root()
    sys.path[:0] = [str(root / "platform/python"), str(root / "library")]
    import torch
    from builders.gfx942.attention.prefill.attention_dense_prefill import (
        dense_request, resolve_dense_spec)
    from kernels.gfx942 import attention_dense as dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig, time_launches
    return locals()


def runtime_args(spec, gpu):
    """Support both older baked-shape and newer runtime-shape kernel ABIs."""
    dense = gpu["dense"]
    helper = getattr(dense, "attention_dense_runtime_args", None)
    if callable(helper):
        return dict(helper(spec))
    names = tuple(getattr(spec, "runtime_kernarg_fields", ()))
    if not names:
        if getattr(spec, "runtime_shape", False):
            raise RuntimeError("Runtime-shape ABI has no accessible kernarg metadata")
        return {}
    values = {n: int(getattr(spec, n)) for n in names if hasattr(spec, n)}
    if set(names) - set(values):
        fastdiv = getattr(dense, "_fastdiv_kernargs", None)
        if callable(fastdiv):
            values.update(fastdiv(spec))
        elif getattr(spec, "fastdiv_divisors", ()):
            from rocke.helpers.transforms import calculate_magic_numbers
            nqb = spec.seqlen_q // spec.block_m
            if nqb * spec.num_query_heads * spec.batch >= (1 << 30):
                raise ValueError("Persistent work exceeds fast-division limit")
            divisors = {"batch": spec.batch, "num_query_heads": spec.num_query_heads,
                        "gqa": spec.num_queries_per_kv, "nqb": nqb}
            for n in spec.fastdiv_divisors:
                magic, shift = calculate_magic_numbers(divisors[n])
                values[n + "_magic"] = magic - (1 << 32) if magic >= (1 << 31) else magic
                values[n + "_shift"] = shift
    missing = set(names) - set(values)
    if missing:
        raise RuntimeError(f"Missing required runtime kernel arguments: {sorted(missing)}")
    return {n: values[n] for n in names}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, sort_keys=True) + "\n")


def git(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def spec_fields(spec):
    return {f.name: getattr(spec, f.name) for f in fields(spec)}


def identity(spec):
    """Canonical effective codegen/grid identity (not raw policy spellings)."""
    return (spec.block_m, spec.block_n, spec.resolved_waves_per_eu(),
            spec.lds_row_pad, spec.resolved_use_exp2_fast(),
            spec.resolved_use_cfvst(),
            spec.resolved_v_row_pad() if spec.resolved_use_cfvst() else None,
            spec.resolved_use_v_swizzle() if spec.resolved_use_cfvst() else False,
            spec.pv_loop_order, spec.pv_priority, spec.o_store_width,
            spec.causal_diag_split, spec.iglp,
            spec.iglp_mode if spec.iglp else 0,
            spec.pv_sched_fence_mask if not spec.iglp else None,
            spec.persistent, spec.num_persistent if spec.persistent else None,
            spec.resolved_persist_decode if spec.persistent else None,
            spec.interleave if spec.persistent else False)


def ensure_baseline_path(spec):
    """Do not accidentally search any part of the optimization under review."""
    if spec.resolved_use_cfvst():
        raise RuntimeError("Production BF16 D128 base resolves CFVST on; "
                           "this benchmark requires a non-CFVST baseline")
    if spec.lds_num_buffers != 1:
        raise RuntimeError("Production base enables K double buffering")
    if getattr(spec, "v_reg_prefetch", False):
        raise RuntimeError("Production base enables explicit V-register prefetch")


def legal_specs(base, variants, dense):
    """Reject unsupported specs and avoid recompiling alias configurations."""
    results, excluded, aliases, seen = [], [], 0, {}
    for knobs in variants:
        unexpected = set(knobs) - SWEEP_KNOBS
        if unexpected:
            raise ValueError(f"Sweep tries to override excluded knobs: {sorted(unexpected)}")
        try:
            spec = replace(base, **knobs)
            # Explicitly prove that all non-sweep fields are still production values.
            for f in fields(base):
                if f.name not in SWEEP_KNOBS and getattr(base, f.name) != getattr(spec, f.name):
                    raise RuntimeError(f"Production-only field altered: {f.name}")
            ensure_baseline_path(spec)
            ok, reason = dense.supports_attention_dense(spec, arch=ARCH)
            if not ok:
                excluded.append({"knobs": knobs, "reason": reason})
                continue
        except (ValueError, TypeError, RuntimeError) as exc:
            excluded.append({"knobs": knobs, "reason": str(exc)})
            continue
        name, signature = dense.gfx942_kernel_name(spec), identity(spec)
        if name in seen:
            if seen[name] != signature:
                raise RuntimeError(f"Kernel-name collision for {name}: {knobs}")
            aliases += 1
            continue
        seen[name] = signature
        results.append((spec, knobs))
    return results, excluded, aliases


class Workload:
    """Scaled BF16 inputs and a chunked independent FP32 attention reference."""
    def __init__(self, base, torch, seed=0, q_chunk=128, head_chunk=4):
        self.torch = torch
        self.scale = 1 / math.sqrt(base.head_size)
        b, s = base.batch, base.seqlen_q
        hq, hkv, d = base.num_query_heads, base.num_kv_heads, base.head_size
        torch.manual_seed(seed)
        self.q, self.k, self.v = [
            (torch.randn(b, s, h, d, dtype=torch.bfloat16, device="cuda") * 0.2)
            .contiguous() for h in (hq, hkv, hkv)]
        self.out = torch.empty_like(self.q)
        self.ref = torch.empty((b, s, hq, d), device="cuda", dtype=torch.float32)
        # Global query indices matter: is_causal=True on a query chunk would
        # incorrectly reset the causal diagonal to zero.
        keys = torch.arange(s, device="cuda")[None, :]
        for bi in range(b):
            for h0 in range(0, hq, head_chunk):
                h1 = min(hq, h0 + head_chunk)
                idx = torch.arange(h0, h1, device="cuda") // (hq // hkv)
                kh = self.k[bi:bi + 1, :, idx].permute(0, 2, 1, 3).float()
                vh = self.v[bi:bi + 1, :, idx].permute(0, 2, 1, 3).float()
                for q0 in range(0, s, q_chunk):
                    q1 = min(s, q0 + q_chunk)
                    q = self.q[bi:bi + 1, q0:q1, h0:h1].permute(0, 2, 1, 3).float()
                    mask = torch.arange(q0, q1, device="cuda")[:, None] >= keys
                    ref = torch.nn.functional.scaled_dot_product_attention(
                        q, kh, vh, attn_mask=mask, scale=self.scale)
                    self.ref[bi:bi + 1, q0:q1, h0:h1] = ref.permute(0, 2, 1, 3)
        torch.cuda.synchronize()

    def max_abs(self):
        # GPU reduction propagates a NaN from ANY batch (Python max() does not).
        errors = [(self.out[b].float() - self.ref[b]).abs().max()
                  for b in range(self.q.shape[0])]
        return self.torch.stack(errors).max().item()


class Runner:
    def __init__(self, gpu, workload):
        self.gpu, self.w = gpu, workload
        self.cache = OrderedDict()

    def prepare(self, spec):
        gpu = self.gpu
        dense = gpu["dense"]
        name = dense.gfx942_kernel_name(spec)
        if name not in self.cache:
            artifact = gpu["compile_kernel"](
                dense.build_attention_dense(spec, arch=ARCH), arch=ARCH,
                backend="python", capture_ir_text=False)
            self.cache[name] = gpu["KernelLauncher"](
                hsaco=artifact.hsaco, kernel_name=artifact.kernel_name,
                signature=dense.attention_dense_signature(spec))
            if len(self.cache) > 24:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        args = {"q_ptr": self.w.q, "k_ptr": self.w.k, "v_ptr": self.w.v,
                "o_ptr": self.w.out, "scale": self.w.scale}
        args.update(runtime_args(spec, gpu))
        config = gpu["LaunchConfig"](
            grid=dense.attention_dense_grid(spec),
            block=dense.attention_dense_block(spec),
            stream=gpu["torch"].cuda.current_stream().cuda_stream)
        launcher = self.cache[name]
        return lambda: launcher(args, config=config)

    def check(self, spec, tol):
        self.w.out.fill_(float("nan"))
        self.prepare(spec)()  # A single launch uses the launcher's safe fence.
        self.gpu["torch"].cuda.synchronize()
        err = self.w.max_abs()
        if not math.isfinite(err):
            return None
        try:
            self.gpu["torch"].testing.assert_close(
                self.w.out.float(), self.w.ref, atol=tol, rtol=tol)
        except AssertionError:
            return None
        return err

    def window(self, spec, warmup, iters):
        fn = self.prepare(spec)
        stream = self.gpu["torch"].cuda.current_stream().cuda_stream
        ms = float(self.gpu["time_launches"](
            fn, warmup=warmup, iters=iters, stream=stream))
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"Invalid timed latency {ms}")
        return ms

    def screen(self, spec, args):
        error = self.check(spec, args.tol)
        if error is None:
            raise ValueError("FP32 numeric gate failed")
        self.window(spec, args.screen_warmup, args.screen_iters)  # discard cold
        times = [self.window(spec, args.screen_warmup, args.screen_iters)
                 for _ in range(args.screen_repeats)]
        return statistics.median(times), error

    def pair(self, baseline, spec, args):
        for s in (baseline, spec):
            self.window(s, args.round_warmup, args.round_iters)
        ratios = []
        for i in range(args.rounds):
            order = (baseline, spec) if i % 2 == 0 else (spec, baseline)
            times = {id(s): self.window(s, args.round_warmup, args.round_iters)
                     for s in order}
            ratios.append(times[id(baseline)] / times[id(spec)])
        return {"ratio": statistics.median(ratios), "min": min(ratios),
                "max": max(ratios), "spread_pct": 100 * (max(ratios) - min(ratios)) /
                statistics.median(ratios)}


def _chosen_for_ab(passed, n, *, preserve_geometry=False):
    """Short screen ranking plus one representative from diverse tile shapes."""
    chosen, seen = [], set()
    if preserve_geometry:
        geom_seen = set()
        for row in passed:
            knobs = row["knobs"]
            pair = (knobs.get("block_m"), knobs.get("block_n"))
            if pair in geom_seen:
                continue
            geom_seen.add(pair)
            key = row["kernel"]
            chosen.append(row)
            seen.add(key)
            if len(chosen) >= min(n, 9):
                break
    for row in passed:
        if len(chosen) >= n:
            break
        if row["kernel"] not in seen:
            chosen.append(row)
            seen.add(row["kernel"])
    return chosen


def _choose_beam(base, dense, seeds, timed, width, *, preserve_geometry=False):
    """Combine all A/B winners with preceding beam; keep a baseline anchor."""
    anchor = {"knobs": {}, "ratio": 1.0, "baseline": True}
    pool = [anchor, *seeds, *timed]
    unique_by_kernel = {}
    for row in pool:
        try:
            k = dense.gfx942_kernel_name(replace(base, **row["knobs"]))
        except (TypeError, ValueError):
            continue
        if k not in unique_by_kernel or row.get("ratio", 0) > unique_by_kernel[k].get("ratio", 0):
            unique_by_kernel[k] = {"knobs": dict(row["knobs"]),
                                   "ratio": float(row.get("ratio", 1.0)),
                                   "baseline": bool(row.get("baseline", False)),
                                   "kernel": k}
    ranked = sorted(unique_by_kernel.values(), key=lambda r: r["ratio"], reverse=True)
    selected, seen, geom_seen = [], set(), set()
    if preserve_geometry:
        # Keep good examples of alternative (block_m, block_n) choices.
        for row in ranked:
            geom = (row["knobs"].get("block_m", base.block_m),
                    row["knobs"].get("block_n", base.block_n))
            if geom not in geom_seen and row["knobs"]:
                selected.append(row)
                seen.add(row["kernel"])
                geom_seen.add(geom)
                if len(selected) >= min(6, max(1, width - 1)):
                    break
    for row in ranked:
        if len(selected) >= width - 1:
            break
        if row["kernel"] not in seen and row["knobs"]:
            selected.append(row)
            seen.add(row["kernel"])
    # Reserve one slot for the production base; it can improve later.
    if any(row["knobs"] == {} for row in selected):
        selected = [r for r in selected if r["knobs"] != {}]
    selected.append(anchor)
    return selected[:width]


def stage(name, base, variants, runner, args, shape_dir, *, seeds=(), beam_width=8):
    """Validate/screen full configurations, A/B shortlist, and propagate a beam."""
    dense = runner.gpu["dense"] if runner else args._dense
    legal, rejected, aliases = legal_specs(base, unique(variants), dense)
    print(f"  {name}: legal={len(legal)}, rejected={len(rejected)}, aliases={aliases}",
          flush=True)
    if args.dry_run:
        # Counts for downstream stages use a representative synthetic beam,
        # never claim that they are the counts of a measured winning beam.
        representatives = [{"knobs": knobs, "ratio": 1.0}
                           for _, knobs in legal[:max(1, min(beam_width, 8))]]
        representatives = representatives[:max(0, beam_width - 1)]
        representatives.append({"knobs": {}, "ratio": 1.0, "baseline": True})
        dump(shape_dir / f"{name}.json", {
            "status": "dry_run", "legal": len(legal), "excluded": rejected,
            "aliases": aliases, "timings_taken": False,
            "note": "downstream search sizes depend on actual measured beam",
        })
        return {"ab": [], "beam": representatives[:beam_width],
                "status": "dry_run", "legal": len(legal),
                "rejected_count": len(rejected), "aliases": aliases}
    screened, retimed = [], []
    base_name = dense.gfx942_kernel_name(base)
    for i, (spec, knobs) in enumerate(legal, 1):
        if dense.gfx942_kernel_name(spec) == base_name:
            continue
        record = {"knobs": knobs, "kernel": dense.gfx942_kernel_name(spec)}
        try:
            ms, err = runner.screen(spec, args)
            record.update(screen_ms=ms, max_abs=err)
        except ValueError as exc:
            if str(exc) == "FP32 numeric gate failed":
                record["invalid_numeric"] = str(exc)
            else:
                record["error"] = str(exc)
        except Exception as exc:
            record["error"] = str(exc)
        screened.append(record)
        if i % 20 == 0 or i == len(legal):
            print(f"    {name}: {i}/{len(legal)} processed", flush=True)
    passed = sorted((r for r in screened if "screen_ms" in r),
                    key=lambda r: r["screen_ms"])
    ab_pool = _chosen_for_ab(passed, max(args.top_per_stage, beam_width),
                             preserve_geometry=(name == "A1"))
    for r in ab_pool:
        spec = replace(base, **r["knobs"])
        try:
            retimed.append({**r, **runner.pair(base, spec, args)})
        except Exception as exc:
            retimed.append({**r, "error": f"A/B: {exc}"})
    ranked = sorted((r for r in retimed if "ratio" in r),
                    key=lambda r: r["ratio"], reverse=True)
    beam = _choose_beam(base, dense, seeds, ranked, beam_width,
                        preserve_geometry=(name == "A1"))
    # Numeric failures are excluded, but reported; infrastructure/compiler errors
    # mark the search incomplete rather than silently inflating its success.
    failed = sum("error" in r for r in screened)
    status = "complete" if not failed and all("error" not in r for r in retimed) \
             else "partial_or_incomplete"
    report = {"status": status, "legal": len(legal), "rejected": rejected,
              "aliases": aliases, "screened": screened, "ab": retimed,
              "beam": beam, "best_screened": passed[0] if passed else None,
              "best_ab": ranked[0] if ranked else None,
              "screen_errors": failed,
              "numeric_rejections": sum("invalid_numeric" in r for r in screened)}
    dump(shape_dir / f"{name}.json", report)
    return {"ab": ranked, "beam": beam, "legal": len(legal),
            "rejected_count": len(rejected), "aliases": aliases,
            "screened": len(screened), "correct_screened": len(passed),
            "ab_tested": len(ranked), "status": status}


def finalists(base, dense, results, cap):
    """Preserve both strong current-beam candidates and earlier stage winners."""
    picks = []
    for stage_result in results.values():
        picks.extend(stage_result["beam"])
    picks.extend(r for stage_result in results.values() for r in stage_result["ab"])
    return [r["knobs"] for r in _choose_beam(base, dense, [], picks,
                                               cap, preserve_geometry=True)]


def verify_finalists(base, gpu, args, picks, shape_dir, runner):
    """Scaled numerical validation and independent A/B final measurement."""
    records = []
    for knobs in picks:
        spec = replace(base, **knobs)
        entry = {"knobs": knobs, "kernel": gpu["dense"].gfx942_kernel_name(spec)}
        try:
            error = runner.check(spec, args.tol)
            entry["max_abs_scaled"] = error
            if error is None:
                raise ValueError("Scaled FP32 numeric gate failed")
            if knobs:
                entry.update(runner.pair(base, spec, args))
            else:
                entry.update({"ratio": 1.0, "min": 1.0, "max": 1.0,
                              "spread_pct": 0.0, "baseline": True})
        except Exception as exc:
            entry["error"] = str(exc)
        records.append(entry)
    valid = sorted((r for r in records if "ratio" in r and r["knobs"]),
                   key=lambda r: r["ratio"], reverse=True)
    best = valid[0] if valid else None
    summary = {"shape": shape_dir.name,
               "status": ("complete" if all("error" not in r for r in records)
                          else "partial_or_incomplete"),
               "baseline_kernel": gpu["dense"].gfx942_kernel_name(base),
               "numeric_profiles": ["scaled"],
               "best_nonbaseline": best,
               "best_nonbaseline_full_spec": spec_fields(replace(base, **best["knobs"])) if best else None,
               "best_overall": best if best and best["ratio"] > 1 else
                               {"ratio": 1.0, "knobs": {}, "baseline": True},
               "tested": len(records), "correct_ab": len(records) - sum("error" in r for r in records)}
    dump(shape_dir / "final_validation.json", {"summary": summary, "candidates": records})
    return summary



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("sweep_results"),
                   help="Result directory; relative paths resolve beside sweep.py")
    p.add_argument("--mode", choices=("baseline", "discover", "all"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Reuse fully completed per-shape results with matching script, kernel, and stack")
    p.add_argument("--batches", default="1,16",
                   help="Filter the fixed 12-shape cohort by batch (1 or 16); defaults to both")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--screen-warmup", type=int, default=5)
    p.add_argument("--screen-iters", type=int, default=20)
    p.add_argument("--screen-repeats", type=int, default=3)
    p.add_argument("--round-warmup", type=int, default=10)
    p.add_argument("--round-iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--top-per-stage", type=int, default=24,
                   help="Maximum A/B screening pool, subject to stage beam minimum")
    p.add_argument("--a1-beam", type=int, default=16)
    p.add_argument("--a2-beam", type=int, default=8)
    p.add_argument("--c-beam", type=int, default=6)
    p.add_argument("--b1-beam", type=int, default=10)
    p.add_argument("--b2-beam", type=int, default=8)
    p.add_argument("--d-beam", type=int, default=8)
    p.add_argument("--refine-seeds", type=int, default=3,
                   help="Top full configs starting each final refinement round")
    p.add_argument("--refine-passes", type=int, default=2,
                   help="Number of bounded local cross-group refinement passes (0 disables)")
    p.add_argument("--finalists", type=int, default=10)
    p.add_argument("--ref-q-chunk", type=int, default=128)
    p.add_argument("--ref-head-chunk", type=int, default=4)
    args = p.parse_args()
    if args.resume and (args.mode != "all" or args.dry_run):
        p.error("--resume requires --mode all without --dry-run")
    if args.dry_run and args.mode == "all":
        p.error("--dry-run requires --mode discover (or baseline)")
    try:
        args.batches = tuple(dict.fromkeys(int(x.strip()) for x in args.batches.split(",")))
    except ValueError:
        p.error("--batches must be comma-separated integers, e.g. 1,16")
    positive = ("top_per_stage", "a1_beam", "a2_beam", "c_beam", "b1_beam",
                "b2_beam", "d_beam", "refine_seeds", "finalists", "ref_q_chunk",
                "ref_head_chunk", "screen_iters", "round_iters", "screen_warmup",
                "round_warmup", "screen_repeats")
    if (not args.batches or any(b not in {1, 16} for b in args.batches)
            or args.rounds < 3 or args.refine_passes not in (0, 1, 2)
            or any(getattr(args, k) < 1 for k in positive)
            or args.tol <= 0):
        p.error("Invalid shape sizes or timing/search configuration")
    if args.finalists < 2:
        p.error("--finalists must be at least 2 to preserve the production baseline")
    for k in ("a1_beam", "a2_beam", "c_beam", "b1_beam", "b2_beam", "d_beam"):
        if getattr(args, k) < 2:
            p.error(f"--{k.replace('_', '-')} must be >=2 to preserve the baseline")
    return args


def try_command(*command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=8,
                                check=False)
        return {"exit_code": result.returncode,
                "stdout": result.stdout.strip()[:1200],
                "stderr": result.stderr.strip()[:300]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def library_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(root, torch, dense, args, shapes):
    # Local provenance. DO NOT publish unreviewed versions or private paths.
    gpu_name, gpu_cus, arch_name = None, None, None
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name()
        gpu_cus = getattr(prop, "multi_processor_count", None)
        arch_name = getattr(prop, "gcnArchName", None)
    script_path = Path(__file__).resolve()
    kernel_path = Path(dense.__file__).resolve()
    repo = root.parents[2] if root.name == "rocke" else root
    # The full tracked diff is not included: it could contain confidential data.
    status = git(root, "status", "--porcelain")
    tracked_diff = git(root, "diff", "--name-only")
    tracked_staged = git(root, "diff", "--cached", "--name-only")
    env_keys = ("ROCKE_BACKEND", "ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB",
                "ROCKE_HIP_LIB", "HIP_PATH", "ROCM_PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "LOCAL, PRIVATE experimental provenance; disclosure review required",
        "scope": {"architecture": ARCH, "shapes": list(shapes), "dtype": DTYPE,
                  "mask": "causal", "search": "bounded staged beam, not exhaustive",
                  "cfvst_swept": False, "v_row_pad_swept": False,
                  "optimization_knobs_overridden": False,
                  "baseline": "gfx942_dense production dispatch; BF16 D128 non-CFVST, K buffers=1",
                  "grid_policy": "production resolved once per shape, not retuned after geometry changes",
                  "stage_order": list(STAGE_NAMES[:6]) +
                                  list(STAGE_NAMES[6:6 + args.refine_passes]),
                  "search_domains": {
                      "A1": {"block_m": list(BLOCK_M), "block_n": list(BLOCK_N),
                             "lds_row_pad": list(LDS_ROW_PAD), "waves_per_eu": list(A1_WPE)},
                      "A2": {"waves_per_eu": list(WPE), "pv_loop_order": list(PV_ORDER),
                             "causal_diag_split": list(DIAG_SPLIT)},
                      "C": {"use_exp2_fast": [None, False, True],
                            "causal_diag_split": list(DIAG_SPLIT), "waves_per_eu": list(WPE)},
                      "B1": {"pv_loop_order": list(PV_ORDER),
                             "pv_priority": list(range(4)),
                             "scheduling": list(scheduling())},
                      "B2": {"waves_per_eu": list(WPE), "pv_loop_order": list(PV_ORDER)},
                      "D": {"o_store_width": list(STORE_WIDTH)},
                      "R": "bounded local single-axis neighborhood of top complete specs",
                  }},
        "source": {
            "git_head": git(root, "rev-parse", "HEAD"),
            "git_branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(status.strip()),
            "git_status_porcelain": status,
            "tracked_changed_paths": tracked_diff,
            "staged_changed_paths": tracked_staged,
            "script_path": str(script_path),
            "script_path_repo_relative": str(script_path.relative_to(repo)) if script_path.is_relative_to(repo) else None,
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "kernel_path_repo_relative": str(kernel_path.relative_to(repo)) if kernel_path.is_relative_to(repo) else None,
            "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        },
        "hardware": {"target_arch": ARCH, "runtime_arch": arch_name,
                      "gpu_name_private": gpu_name, "cus": gpu_cus},
        "software": {"python": sys.version, "python_executable": sys.executable,
                     "os_platform": platform.platform(),
                     "torch": torch.__version__, "torch_hip": getattr(torch.version,"hip",None),
                     "triton": library_version("triton"),
                     "rocm_core_wheel": library_version("rocm-sdk-core"),
                     "llvm_flavor": os.environ.get("ROCKE_LLVM_FLAVOR", "UNSET"),
                     "compile_backend": "python",
                     "hipcc_version": try_command("hipcc", "--version"),
                     "hipconfig_version": try_command("hipconfig", "--version"),
                     "clang_version": try_command("clang", "--version")},
        "library_environment_PRIVATE": {k: os.environ[k] for k in env_keys if k in os.environ},
        "invocation": {"cwd": str(Path.cwd()), "argv": sys.argv,
                       "command": shlex.join([sys.executable, str(script_path), *sys.argv[1:]])},
        "tolerance": {"dtype": DTYPE, "atol": args.tol, "rtol": args.tol,
                      "method": "torch.testing.assert_close(out.float(), independent_chunked_fp32_sdpa)"},
        "timing": {"clock": "HIP device events via rocke.runtime.time_launches",
                   "screen_warmup": args.screen_warmup,
                   "screen_iters": args.screen_iters,
                   "screen_repeats": args.screen_repeats,
                   "round_warmup": args.round_warmup,
                   "round_iters": args.round_iters,
                   "rounds": args.rounds,
                   "ratio_statistic": "median of alternating paired A/B round ratios"},
        "parameters": {k:v for k,v in vars(args).items() if not k.startswith("_")},
        "validation_input": {"distribution": "torch.randn", "scale_each_QKV": 0.2,
                             "seed": args.seed, "reference_q_chunk": args.ref_q_chunk,
                             "reference_head_chunk": args.ref_head_chunk},
    }
    return data


def save_replay(out, env):
    invocation = env["invocation"]
    selected = ("ROCKE_BACKEND", "ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB",
                "ROCKE_HIP_LIB", "HIP_PATH", "ROCM_PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             "# LOCAL ONLY: generated paths and library versions may require disclosure review",
             "cd " + shlex.quote(invocation["cwd"])]
    for key in selected:
        value = env["library_environment_PRIVATE"].get(key)
        if value:
            lines.append("export " + key + "=" + shlex.quote(value))
    lines.append(invocation["command"])
    path = out / "replay_command.sh"
    path.write_text("\n".join(lines)+"\n")
    path.chmod(0o755)


def write_relative_report(out, env, rows):
    # Relative-only public format; keep LOCAL DRAFT pending review.
    shape_spec = env["scope"]
    s = ["# gfx942 BF16 D128 causal attention: bounded lever sweep results", "",
         "**LOCAL DRAFT — review software versions and source provenance before publishing.**", "",
         f"Measured scope: gfx942, {env['hardware']['cus']} CUs; {len(shape_spec['shapes'])} "
         "requested BF16 D128 causal GQA shapes; production-resolved gfx942_dense baseline.",
         f"Source git HEAD: `{env['source']['git_head']}`; working tree dirty: "
         f"`{env['source']['git_dirty']}`; LLVM flavor: "
         f"`{env['software']['llvm_flavor']}`.", "",
          "Search: A1 geometry / A2 geometry interactions / C softmax and causal / "
          "B1 scheduling / B2 scheduling-register interaction / D output stores / "
          "bounded final refinement; **not exhaustive**. All candidates retain the "
          "production-resolved non-CFVST, single-K-buffer path; CFVST/KDB/VPF and "
          "grid policy were NOT swept.", "",
         "Independent correctness: BF16 Q/K/V each drawn from torch.randn times 0.2 "
         "against chunked FP32 causal SDPA, "
         f"`torch.testing.assert_close(atol=rtol={env['tolerance']['atol']})`; "
         "failed candidates excluded. Finalists are validated again before A/B timing.",
         "Timing: HIP event timing, screened with short windows; "
         "best finalists compared by alternating baseline/candidate order in the same "
         f"session for {env['timing']['rounds']} rounds of "
         f"{env['timing']['round_iters']} timed launches, with "
         "median paired-round ratio and minimum/maximum round ratios.", "",
         "Detailed local files: `provenance.json` (stack/replay/git state), "
         "`replay_command.sh`, per-shape `baseline.json` and `final_validation.json`, "
         "`manifest.json`.", "",
         "## Best A/B-tested non-baseline finalist per shape", "",
         "| B | S | Hq | Hkv | Best speedup vs baseline | Paired min..max | Best tested knobs | Status |",
         "|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        shape = row["shape"]
        # Shape uses existing B#_S#_Hq#_Hkv#_D# convention.
        vals = [int(part[len(prefix):]) for part, prefix in zip(shape.split('_'), ('B','S','Hq','Hkv','D'))]
        b, seq, hq, hkv, _d = vals
        best = row.get("best_nonbaseline")
        if best:
            ratio = f"{best['ratio']:.5f}x"
            spread = f"{best['min']:.5f}x .. {best['max']:.5f}x"
            knobs = "`" + json.dumps(best['knobs'], sort_keys=True) + "`"
        else:
            ratio, spread, knobs = "not measured", "—", "—"
        s.append(f"| {b} | {seq} | {hq} | {hkv} | {ratio} | {spread} | {knobs} | {row['status']} |")
    s += ["", "**Interpretation:** A value below 1.0x is a regression. "
          "Only configurations that reached final A/B remeasurement can be ranked; "
          "the selected winner is not guaranteed to be the global best.", "",
          "This is not evidence about additional GPU parts, toolchain flavors, "
          "or the opt-in CFVST pipeline."]
    (out / "lever_sweep_results.md").write_text("\n".join(s)+"\n")


def _checked_resume_provenance(out, env):
    """Reject stale/incompatible results before reusing per-shape measurements."""
    prov_path, manifest_path = out / "provenance.json", out / "manifest.json"
    if not prov_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("--resume needs provenance.json and manifest.json from the prior run")
    old = json.loads(prov_path.read_text())
    old_manifest = json.loads(manifest_path.read_text())
    checks = (
        ("git SHA", old.get("source", {}).get("git_head"), env["source"]["git_head"]),
        ("kernel SHA256", old.get("source", {}).get("kernel_sha256"), env["source"]["kernel_sha256"]),
        ("sweep script SHA256", old.get("source", {}).get("script_sha256"), env["source"]["script_sha256"]),
        ("LLVM flavor", old.get("software", {}).get("llvm_flavor"), env["software"]["llvm_flavor"]),
        ("scope", old.get("scope", {}).get("shapes"),
         json.loads(json.dumps(env["scope"]["shapes"]))),
    )
    for label, prev, current in checks:
        if prev != current:
            raise RuntimeError(f"--resume incompatible {label}: previous={prev!r}; now={current!r}")
    # --resume is a new script flag, not a benchmark parameter.
    old_params = {k:v for k,v in old_manifest.get("parameters", {}).items() if k != "resume"}
    new_params = {k:v for k,v in json.loads(json.dumps(env["parameters"], default=str)).items()
                  if k != "resume"}
    if old_params != new_params:
        delta = {k: (old_params.get(k), new_params.get(k))
                 for k in sorted(set(old_params) | set(new_params))
                 if old_params.get(k) != new_params.get(k)}
        raise RuntimeError(f"--resume benchmark parameters differ: {delta}")
    if old.get("source", {}).get("git_dirty") or env["source"].get("git_dirty"):
        print("WARNING: --resume with a dirty Git tree: SHA + kernel-file hash "
              "cannot prove every imported source file is unchanged", flush=True)
    print("Resume provenance: same revision, kernel hash, LLVM flavor, shape cohort, "
          "and benchmark parameters", flush=True)


def _completed_shape_result(shape_dir, kernel_name, refine_passes):
    """Reuse a shape only if every required stage and final validation completed."""
    path = shape_dir / "final_validation.json"
    if not path.is_file():
        return None
    obj = json.loads(path.read_text())
    summary = obj.get("summary", {})
    if (summary.get("shape") != shape_dir.name
            or summary.get("status") != "complete"
            or summary.get("baseline_kernel") != kernel_name
            or summary.get("correct_ab", 0) < 1):
        return None
    stages = {}
    for name in (*STAGE_NAMES[:6], *STAGE_NAMES[6:6 + refine_passes]):
        stage_path = shape_dir / f"{name}.json"
        if not stage_path.is_file():
            return None
        result = json.loads(stage_path.read_text())
        if result.get("status") != "complete":
            return None
        stages[name] = result
    summary["stage_statuses"] = {name: "complete" for name in stages}
    summary["stage_counts"] = {
        name: {"legal": x.get("legal"),
               "rejected_count": len(x.get("rejected", [])),
               "aliases": x.get("aliases"),
               "screened": len(x.get("screened", [])),
               "correct_screened": sum("screen_ms" in r for r in x.get("screened", [])),
               "ab_tested": sum("ratio" in r for r in x.get("ab", []))}
        for name, x in stages.items()}
    return summary


def main():
    args = parse_args()
    gpu = load()
    dense, torch = gpu["dense"], gpu["torch"]
    torch.backends.cuda.matmul.allow_tf32 = False
    args._dense = dense
    root = find_root()
    script_dir = Path(__file__).resolve().parent
    out = (args.out_dir if args.out_dir.is_absolute()
           else script_dir / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Do not accidentally add raw performance results or private stack info to Git.
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    shapes = tuple(shape for shape in COHORT_SHAPES if shape[0] in args.batches)
    print(f"Selected {len(shapes)} of {len(COHORT_SHAPES)} cohort shapes:", flush=True)
    for b, s, hq, hkv, d in shapes:
        print(f"  B={b} S={s} Hq={hq} Hkv={hkv} D={d} dtype={DTYPE}", flush=True)
    env = capture_environment(root, torch, dense, args, shapes)
    if args.resume:
        _checked_resume_provenance(out, env)
    dump(out / "provenance.json", env)
    save_replay(out, env)
    print("Source:", env["source"]["git_head"], "dirty=", env["source"]["git_dirty"],
          "| gpu=", ARCH, "cus=", env["hardware"]["cus"],
          "| flavor=", env["software"]["llvm_flavor"], flush=True)
    print("PROVENANCE:", out / "provenance.json", flush=True)
    print("CAUTION: non-exhaustive 12-knob baseline-only sweep; no CFVST/KDB/VPF arms", flush=True)
    if env["software"]["llvm_flavor"] == "UNSET":
        print("WARNING: ROCKE_LLVM_FLAVOR unset; record actual compiler flavor", flush=True)
    manifest = {"git_sha": git(root, "rev-parse", "HEAD"),
                "git_status_tracked": git(root, "status", "--porcelain", "--untracked-files=no"),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "shapes": list(shapes), "mode": args.mode, "tol": args.tol,
                "parameters": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "search": "staged A1/A2/C/B1/B2/D/R beam; best found, not global optimum"}
    dump(out / "manifest.json", manifest)
    report = []
    widths = {"A1": args.a1_beam, "A2": args.a2_beam,
              "C": args.c_beam, "B1": args.b1_beam,
              "B2": args.b2_beam, "D": args.d_beam}
    for batch, seq, hq, hkv, d in shapes:
        shape = f"B{batch}_S{seq}_Hq{hq}_Hkv{hkv}_D{d}"
        shape_dir = out / shape
        req = gpu["dense_request"](argparse.Namespace(use_sinks=False),
            batch=batch, seqlen_q=seq, seqlen_kv=seq, num_query_heads=hq,
            num_kv_heads=hkv, head_size=d, causal=True, dtype=DTYPE)
        base = gpu["resolve_dense_spec"](req, {})
        ok, reason = dense.supports_attention_dense(base, arch=ARCH)
        if not ok:
            raise RuntimeError(f"Unsupported dispatch baseline for {shape}: {reason}")
        ensure_baseline_path(base)
        dump(shape_dir / "baseline.json", {
            "spec": spec_fields(base),
            "resolved": {"persistent": base.persistent,
                         "num_persistent": base.num_persistent,
                         "exp2_fast": base.resolved_use_exp2_fast(),
                         "waves_per_eu": base.resolved_waves_per_eu(),
                         "persist_decode": base.resolved_persist_decode,
                         "use_cfvst": base.resolved_use_cfvst(),
                         "lds_num_buffers": base.lds_num_buffers},
            "kernel_name": dense.gfx942_kernel_name(base),
            "source": "production-dispatch gfx942_dense"})
        print(f"\n{shape}: {dense.gfx942_kernel_name(base)}", flush=True)
        if args.resume:
            old = _completed_shape_result(shape_dir, dense.gfx942_kernel_name(base),
                                          args.refine_passes)
            if old is not None:
                report.append(old)
                best = old.get("best_nonbaseline")
                print((f"  RESUMED: {best['ratio']:.4f}x best non-baseline" if best
                       else "  RESUMED: verified complete"), flush=True)
                continue
        if args.mode == "baseline":
            continue
        runner = None if args.dry_run else Runner(gpu, Workload(
            base, torch, args.seed, q_chunk=args.ref_q_chunk,
            head_chunk=args.ref_head_chunk))
        if runner and runner.check(base, args.tol) is None:
            raise RuntimeError(f"Baseline fails FP32 correctness for {shape}")
        results = {}
        seeds = [{"knobs": {}, "ratio": 1.0, "baseline": True}]
        for name in STAGE_NAMES[:6]:
            results[name] = stage(name, base, stage_variants(name, seeds), runner,
                                  args, shape_dir, seeds=seeds,
                                  beam_width=widths[name])
            seeds = results[name]["beam"]
        for round_i in range(args.refine_passes):
            name = f"R{round_i + 1}"
            # The baseline anchor always survives; refine the best full specs.
            selected = sorted((r for r in seeds if r["knobs"]),
                              key=lambda r: r.get("ratio", 0), reverse=True)
            selected = selected[:args.refine_seeds]
            if not selected:
                selected = [{"knobs": {}, "ratio": 1.0}]
            results[name] = stage(name, base, refinement_variants(selected), runner,
                                  args, shape_dir, seeds=seeds,
                                  beam_width=args.d_beam)
            seeds = results[name]["beam"]
        if args.dry_run:
            continue
        picks = finalists(base, dense, results, args.finalists)
        dump(shape_dir / "shortlist.json", picks)
        if args.mode == "all":
            result = verify_finalists(base, gpu, args, picks, shape_dir, runner)
            result["stage_statuses"] = {n: r["status"] for n, r in results.items()}
            result["stage_counts"] = {
                n: {k: r.get(k) for k in ("legal", "rejected_count", "aliases",
                                         "screened", "correct_screened", "ab_tested")}
                for n, r in results.items()}
            if any(r["status"] != "complete" for r in results.values()):
                result["status"] = "partial_or_incomplete"
            report.append(result)
            best = result["best_nonbaseline"]
            print((f"  best non-baseline: {best['ratio']:.4f}x" if best
                   else "  NO VALID ALTERNATIVE"), flush=True)
        else:
            scored = [r for stage_result in results.values() for r in stage_result["ab"]]
            best = max(scored, key=lambda x: x["ratio"], default=None)
            states = {n: r["status"] for n, r in results.items()}
            report.append({"shape": shape, "best_ab_discovery": best,
                           "stage_statuses": states,
                           "status": "complete" if all(v == "complete" for v in states.values())
                           else "partial_or_incomplete"})
    if report:
        dump(out / "cohort_ratios_only.json", report)
        if args.mode == "all":
            write_relative_report(out, env, report)
            print("Local relative-only report:", out / "lever_sweep_results.md")
    print(f"\nResults: {out}")
    return 0 if all(r.get("status", "complete") == "complete" for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())