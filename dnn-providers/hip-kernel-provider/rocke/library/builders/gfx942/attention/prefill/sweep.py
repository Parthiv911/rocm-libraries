#!/usr/bin/env python3
"""Bounded gfx942 BF16 D128 causal dense-attention lever sweep.

Six default shapes: B=(1,16), S=(1024,2048,4096), Hq/Hkv=32/8.
The non-CFVST, single-buffer baseline is always dispatch-resolved.

Run from rocke/library/builders/gfx942/attention/prefill/:
  python sweep.py --mode baseline --out-dir sweep_results
  python sweep.py --mode discover --dry-run --out-dir sweep_results
  python sweep.py --mode all --out-dir sweep_results

Every legal candidate is checked against a chunked FP32 SDPA reference before
short GPU-event timing. Top candidates receive alternating same-session A/B;
finalists are rechecked on the same scaled inputs before final A/B timing.
Reports identify the best *tested* non-baseline candidate, including regressions.
Not exhaustive.
"""
from __future__ import annotations

import argparse
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
import time

ARCH, H_Q, H_KV, D, DTYPE = "gfx942", 32, 8, 128, "bf16"
SEQS = (1024, 2048, 4096)
FENCES = (0,) + tuple(1 << i for i in range(11)) + (0x108,)
GEOMETRY = {
    "block_m": (128, 256), "block_n": (32, 64),
    "waves_per_eu": (1, 2, 3, 4), "lds_row_pad": (4, 8, 16),
    "use_exp2_fast": (None, False, True),
}


def scheduling():
    return ([{"iglp": False, "iglp_mode": 0, "pv_sched_fence_mask": None}]
            + [{"iglp": False, "iglp_mode": 0, "pv_sched_fence_mask": f}
               for f in FENCES]
            + [{"iglp": True, "iglp_mode": m, "pv_sched_fence_mask": None}
               for m in (0, 1)])


def product(axes):
    for values in itertools.product(*axes.values()):
        yield dict(zip(axes, values))


def ofat():
    yield {}
    axes = dict(pv_loop_order=("d_major", "k_major"), pv_priority=(0, 1, 2, 3),
                o_store_width=(1, 2, 4), causal_diag_split=(False, True), **GEOMETRY)
    for key, values in axes.items():
        for value in values:
            yield {key: value}
    yield from scheduling()


def pv():
    for order, prio, sched in itertools.product(
            ("d_major", "k_major"), range(4), scheduling()):
        yield {"pv_loop_order": order, "pv_priority": prio, **sched}


def outputs():
    yield from product({"o_store_width": (1, 2, 4),
                        "causal_diag_split": (False, True)})


def find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "library").is_dir() and (p / "platform/python").is_dir():
            return p
    raise RuntimeError("Place sweep.py inside a rocke/library checkout")


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
    """Values expected to affect the kernel body or launch identity."""
    return (spec.block_m, spec.block_n, spec.resolved_waves_per_eu(),
            spec.lds_row_pad, spec.resolved_use_exp2_fast(), spec.pv_loop_order,
            spec.pv_priority, spec.o_store_width, spec.causal_diag_split,
            spec.iglp, spec.iglp_mode if spec.iglp else 0,
            spec.pv_sched_fence_mask if not spec.iglp else None,
            spec.persistent, spec.num_persistent if spec.persistent else None,
            spec.resolved_persist_decode if spec.persistent else None,
            spec.interleave if spec.persistent else False)


def legal_specs(base, variants, dense):
    """Prune unsupported variants and canonical name aliases."""
    results, excluded, aliases, seen = [], [], 0, {}
    for knobs in variants:
        try:
            spec = replace(base, **knobs)
            ok, reason = dense.supports_attention_dense(spec, arch=ARCH)
            if not ok or spec.resolved_use_cfvst() or spec.lds_num_buffers != 1:
                excluded.append({"knobs": knobs, "reason": reason if not ok
                                 else "outside non-CFVST single-buffer scope"})
                continue
        except Exception as exc:
            excluded.append({"knobs": knobs, "reason": repr(exc)})
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
        self.scale = 1 / math.sqrt(D)
        b, s = base.batch, base.seqlen_q
        torch.manual_seed(seed)
        self.q, self.k, self.v = [
            (torch.randn(b, s, h, D, dtype=torch.bfloat16, device="cuda") * 0.2)
            .contiguous() for h in (H_Q, H_KV, H_KV)]
        self.out = torch.empty_like(self.q)
        self.ref = torch.empty((b, s, H_Q, D), device="cuda", dtype=torch.float32)
        # Global query indices matter: is_causal=True on a query chunk would
        # incorrectly reset the causal diagonal to zero.
        keys = torch.arange(s, device="cuda")[None, :]
        for bi in range(b):
            for h0 in range(0, H_Q, head_chunk):
                h1 = min(H_Q, h0 + head_chunk)
                idx = torch.arange(h0, h1, device="cuda") // (H_Q // H_KV)
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
        return err if math.isfinite(err) and err <= tol else None

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


def stage(name, base, knobs, runner, args, shape_dir):
    dense = runner.gpu["dense"] if runner else args._dense
    legal, rejected, aliases = legal_specs(base, knobs, dense)
    print(f"  {name}: legal={len(legal)}, rejected={len(rejected)}, aliases={aliases}",
          flush=True)
    if args.dry_run:
        return {"ab": [], "best_screen": None, "status": "dry_run"}
    screened, retimed = [], []
    base_name = dense.gfx942_kernel_name(base)
    for i, (spec, knobs) in enumerate(legal, 1):
        if dense.gfx942_kernel_name(spec) == base_name:
            continue
        record = {"knobs": knobs, "kernel": dense.gfx942_kernel_name(spec)}
        try:
            ms, err = runner.screen(spec, args)
            record.update(screen_ms=ms, max_abs=err)
        except Exception as exc:
            record["error"] = str(exc)
        screened.append(record)
        if i % 20 == 0 or i == len(legal):
            print(f"    {name}: {i}/{len(legal)} processed", flush=True)
    passed = sorted((r for r in screened if "screen_ms" in r),
                    key=lambda r: r["screen_ms"])
    for r in passed[:args.top_per_stage]:
        spec = replace(base, **r["knobs"])
        try:
            retimed.append({**r, **runner.pair(base, spec, args)})
        except Exception as exc:
            retimed.append({**r, "error": f"A/B: {exc}"})
    ranked = sorted((r for r in retimed if "ratio" in r),
                    key=lambda r: r["ratio"], reverse=True)
    # Numeric rejection is normal pruning; errors and absent A/B results are not.
    status = "complete" if ranked and all(
        "error" not in r for r in screened + retimed) else "partial_or_incomplete"
    report = {"status": status, "legal": len(legal), "rejected": rejected,
              "aliases": aliases, "screened": screened, "ab": retimed,
              "best_screened": passed[0] if passed else None,
              "best_ab": ranked[0] if ranked else None}
    dump(shape_dir / f"{name}.json", report)
    return {"ab": ranked, "best_screen": passed[0] if passed else None,
            "status": status}


def unique(items):
    seen = set()
    for knobs in items:
        key = json.dumps(knobs, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield knobs


def interactions(pv_rows, geom_rows, output_rows, n):
    a = [r["knobs"] for r in pv_rows[:n]]
    b = [r["knobs"] for r in geom_rows[:n]]
    combos = [dict(x, **y) for x, y in itertools.product(a, b)]
    if output_rows:
        best = output_rows[0]["knobs"]
        combos += [dict(x, **best) for x in combos[:n]]
    return list(unique(combos))


def finalists(results, cap):
    """Keep each stage's best, then globally strongest A/B-tested alternatives."""
    choices = [r["ab"][0]["knobs"] for r in results.values() if r["ab"]]
    pool = sorted((r for stage_result in results.values()
                   for r in stage_result["ab"]),
                  key=lambda r: r["ratio"], reverse=True)
    return list(unique(choices + [r["knobs"] for r in pool]))[:cap]


def verify_finalists(base, gpu, args, picks, shape_dir, runner):
    """Revalidate scaled inputs, then independently remeasure finalists."""
    records = []
    for knobs in picks:
        spec = replace(base, **knobs)
        entry = {"knobs": knobs, "kernel": gpu["dense"].gfx942_kernel_name(spec)}
        try:
            error = runner.check(spec, args.tol)
            entry["max_abs_scaled"] = error
            if error is None:
                raise ValueError("Scaled FP32 numeric gate failed")
            entry.update(runner.pair(base, spec, args))
        except Exception as exc:
            entry["error"] = str(exc)
        records.append(entry)
    valid = sorted((r for r in records if "ratio" in r),
                   key=lambda r: r["ratio"], reverse=True)
    best = valid[0] if valid else None
    summary = {"shape": shape_dir.name,
               "status": "complete" if valid else "incomplete_no_valid_finalists",
               "baseline_kernel": gpu["dense"].gfx942_kernel_name(base),
               "numeric_profiles": ["scaled"],
               "best_nonbaseline": best,
               "best_overall": best if best and best["ratio"] > 1 else
                               {"ratio": 1.0, "knobs": {}, "baseline": True},
               "tested": len(records), "correct_ab": len(valid)}
    dump(shape_dir / "final_validation.json", {"summary": summary, "candidates": records})
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("baseline", "discover", "all"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batches", default="1,16", help="e.g. 1 or 1,16")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--screen-warmup", type=int, default=5)
    p.add_argument("--screen-iters", type=int, default=20)
    p.add_argument("--screen-repeats", type=int, default=3)
    p.add_argument("--round-warmup", type=int, default=10)
    p.add_argument("--round-iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--top-per-stage", type=int, default=8)
    p.add_argument("--interaction-top", type=int, default=5)
    p.add_argument("--finalists", type=int, default=10)
    p.add_argument("--ref-q-chunk", type=int, default=128)
    p.add_argument("--ref-head-chunk", type=int, default=4)
    args = p.parse_args()
    try:
        args.batches = tuple(dict.fromkeys(int(x.strip()) for x in args.batches.split(",")))
    except ValueError:
        p.error("--batches must be comma-separated integers, e.g. 1,16")
    if (not args.batches or any(b < 1 for b in args.batches)
            or args.rounds < 3 or args.screen_repeats < 1
            or min(args.top_per_stage, args.finalists, args.ref_q_chunk,
                   args.ref_head_chunk, args.screen_iters, args.round_iters) < 1):
        p.error("Invalid batch sizes or measurement configuration")
    return args


def main():
    args = parse_args()
    gpu = load()
    dense, torch = gpu["dense"], gpu["torch"]
    args._dense = dense
    root = find_root()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"git_sha": git(root, "rev-parse", "HEAD"),
                "git_status_tracked": git(root, "status", "--porcelain", "--untracked-files=no"),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "shapes": [(b, s, H_Q, H_KV, D) for b in args.batches for s in SEQS],
                "mode": args.mode, "tol": args.tol,
                "parameters": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "search": "bounded; not exhaustive; best among A/B-tested finalists"}
    dump(out / "manifest.json", manifest)
    report = []
    for batch, seq in itertools.product(args.batches, SEQS):
        shape = f"B{batch}_S{seq}_Hq{H_Q}_Hkv{H_KV}_D{D}"
        shape_dir = out / shape
        req = gpu["dense_request"](argparse.Namespace(use_sinks=False),
            batch=batch, seqlen_q=seq, seqlen_kv=seq, num_query_heads=H_Q,
            num_kv_heads=H_KV, head_size=D, causal=True, dtype=DTYPE)
        base = gpu["resolve_dense_spec"](req, {})
        ok, reason = dense.supports_attention_dense(base, arch=ARCH)
        if not ok or base.resolved_use_cfvst() or base.lds_num_buffers != 1:
            raise RuntimeError(f"Unsupported dispatch baseline for {shape}: {reason}")
        dump(shape_dir / "baseline.json", spec_fields(base))
        print(f"\n{shape}: {dense.gfx942_kernel_name(base)}", flush=True)
        if args.mode == "baseline":
            continue
        runner = None if args.dry_run else Runner(gpu, Workload(base, torch, args.seed,
            q_chunk=args.ref_q_chunk, head_chunk=args.ref_head_chunk))
        if runner and runner.check(base, args.tol) is None:
            raise RuntimeError(f"Baseline fails FP32 correctness for {shape}")
        results = {}
        for name, variants in (("ofat", ofat), ("pv", pv),
                               ("output", outputs), ("geometry", lambda: product(GEOMETRY))):
            results[name] = stage(name, base, variants(), runner, args, shape_dir)
        if args.dry_run:
            continue
        combos = interactions(results["pv"]["ab"], results["geometry"]["ab"],
                              results["output"]["ab"], args.interaction_top)
        results["interaction"] = stage("interaction", base, combos,
                                        runner, args, shape_dir)
        picks = finalists(results, args.finalists)
        dump(shape_dir / "shortlist.json", picks)
        if args.mode == "all":
            result = verify_finalists(base, gpu, args, picks, shape_dir, runner)
            report.append(result)
            best = result["best_nonbaseline"]
            print(f"  best non-baseline: {best['ratio']:.4f}x" if best
                  else "  NO VALID ALTERNATIVE", flush=True)
        else:
            scored = [r for stage_result in results.values() for r in stage_result["ab"]]
            best = max(scored, key=lambda x: x["ratio"], default=None)
            stage_states = {n: r["status"] for n, r in results.items()}
            report.append({"shape": shape, "best_ab_discovery": best,
                           "stage_statuses": stage_states,
                           "status": "complete" if best and all(
                               x == "complete" for x in stage_states.values())
                           else "partial_or_incomplete"})
    if report:
        dump(out / "cohort_ratios_only.json", report)
    print(f"\nResults: {out}")
    return 0 if all(r.get("status", "complete") == "complete" for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())