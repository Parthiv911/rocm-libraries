#!/usr/bin/env python3
"""Three-arm gfx942 comparison for the SIX already-selected lever-sweep winners.

For each shape: (1) production dispatch, (2) exact winning lever settings,
(3) those SAME winning settings with CFVST + 2 K LDS buffers + V register prefetch.

The B=1,S=1024 winner has causal_diag_split=True, incompatible with BF16 CFVST.
We measure its first two arms, mark the third INCOMPATIBLE, and DO NOT silently
modify the winning configuration. No new sweep/selection is performed.

Runs directly from rocke/library/benchmarks/gfx942/attention/prefill/rocke_vs_aiter.
Local output includes private absolute GPU timings. Do not commit raw timings.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


# Verbatim winning knob dictionaries from the user's six-shape sweep report.
WINNERS = {
    (1, 1024): {
        "block_m": 128, "block_n": 64, "causal_diag_split": True,
        "iglp": False, "iglp_mode": 0, "lds_row_pad": 4,
        "o_store_width": 4, "pv_loop_order": "k_major", "pv_priority": 1,
        "pv_sched_fence_mask": 4, "use_exp2_fast": True,
        "waves_per_eu": 4,
    },
    (1, 2048): {
        "block_m": 256, "block_n": 64, "lds_row_pad": 4,
        "use_cfvst": True, "use_exp2_fast": True, "waves_per_eu": 4,
    },
    (1, 4096): {"lds_row_pad": 4, "use_cfvst": True},
    (16, 1024): {
        "block_m": 256, "block_n": 64, "lds_row_pad": 4,
        "use_cfvst": True, "use_exp2_fast": None, "waves_per_eu": 1,
    },
    (16, 2048): {
        "block_m": 256, "block_n": 64, "lds_row_pad": 4,
        "use_cfvst": True, "use_exp2_fast": None, "waves_per_eu": 2,
    },
    (16, 4096): {"lds_row_pad": 4, "use_cfvst": True},
}


# Self-contained chunked independent FP32 causal GQA attention.
def fp32_reference(torch, q, k, v, chunk=128):
    batch, seqlen, hq, dim = q.shape
    hkv = k.shape[2]
    group = hq // hkv
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3)
    vf = v.float().permute(0, 2, 1, 3)
    result = torch.empty((batch, hq, seqlen, dim), device=q.device,
                         dtype=torch.float32)
    keys = torch.arange(seqlen, device=q.device).reshape(1, 1, 1, seqlen)
    scale = 1 / math.sqrt(dim)
    with torch.no_grad():
        for kv in range(hkv):
            kh = kf[:, kv:kv + 1].transpose(-1, -2)
            vh = vf[:, kv:kv + 1]
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                scores = torch.matmul(qf[:, kv * group:(kv + 1) * group, qs:qe], kh)
                scores *= scale
                qi = torch.arange(qs, qe, device=q.device).reshape(1, 1, -1, 1)
                scores.masked_fill_(keys > qi, float("-inf"))
                result[:, kv * group:(kv + 1) * group, qs:qe] = torch.matmul(
                    torch.softmax(scores, dim=-1), vh
                )
    return result.permute(0, 2, 1, 3).contiguous()


def find_rocke():
    starts = [Path(__file__).resolve().parent, Path.cwd()]
    if os.environ.get("ROCKE_ROOT"):
        starts.insert(0, Path(os.environ["ROCKE_ROOT"]))
    for start in starts:
        for parent in (start, *start.parents):
            for root in (parent, parent / "dnn-providers/hip-kernel-provider/rocke"):
                if (root / "library/kernels/gfx942/attention_dense.py").is_file():
                    return root.resolve()
    raise RuntimeError("Cannot locate ROCKE; place this script in rocke_vs_aiter or set ROCKE_ROOT")


def git(root, *command):
    try:
        return subprocess.check_output(["git", *command], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def timed(torch, launcher, values, config, stream, repeat):
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    first.record(stream)
    for _ in range(repeat):
        launcher(values, config=config)
    last.record(stream)
    last.synchronize()
    ms = float(first.elapsed_time(last)) / repeat
    if not math.isfinite(ms) or ms <= 0:
        raise RuntimeError(f"Invalid timing {ms}")
    return ms


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape", choices=[f"B{b}_S{s}" for b, s in WINNERS],
                   action="append", help="Optional subset; can repeat")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--list", action="store_true", help="Show the six winning specs without GPU execution")
    args = p.parse_args()
    if args.rounds < 1 or args.repeat < 1 or args.warmup < 0:
        p.error("--rounds and --repeat must be >=1; --warmup must be >=0")
    selected = [(b, s) for b, s in WINNERS if args.shape is None or f"B{b}_S{s}" in args.shape]
    if args.list:
        for b, s in selected:
            print(f"B={b}, S={s}: winner overrides={WINNERS[b, s]}")
            if (b, s) == (1, 1024):
                print("  Combined optimization: incompatible (causal_diag_split=True)")
        return 0

    rocke = find_rocke()
    for path in (rocke / "library", rocke / "platform/python", rocke):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    from dispatch.attention import AttentionRequest, attention_tuning_spec
    from kernels.gfx942 import attention_dense as dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    if not torch.cuda.is_available():
        raise RuntimeError("No HIP GPU found via torch.cuda")
    expected = (rocke / "library/kernels/gfx942/attention_dense.py").resolve()
    if Path(dense.__file__).resolve() != expected:
        raise RuntimeError(f"Wrong kernel module loaded: {dense.__file__} (expected {expected})")
    required = {"use_cfvst", "lds_num_buffers", "v_reg_prefetch"}
    actual = set(dense.Gfx942AttentionDenseSpec.__dataclass_fields__)
    if not required <= actual:
        raise RuntimeError(f"Kernel source missing independent optimization knobs: {required - actual}")
    if not hasattr(dense, "attention_dense_runtime_args"):
        raise RuntimeError("This runtime-shape kernel requires attention_dense_runtime_args")

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = torch.cuda.current_stream(device=device)
    stream_id = int(stream.cuda_stream)
    rng = random.Random(args.seed)
    out = (args.out_dir or (Path(__file__).resolve().parent / "three_run_results")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    git_sha = git(rocke, "rev-parse", "HEAD")
    git_status = git(rocke, "status", "--porcelain", "--untracked-files=no")
    source_sha = hashlib.sha256(expected.read_bytes()).hexdigest()
    info = dict(git_sha=git_sha, tracked_worktree_dirty=bool(git_status),
                kernel_sha256=source_sha, python=sys.version.split()[0],
                pytorch=torch.__version__, torch_hip=getattr(torch.version, "hip", None),
                llvm_flavor=os.environ.get("ROCKE_LLVM_FLAVOR", "UNSET"),
                gpu_name=torch.cuda.get_device_name(device),
                run=dict(warmup=args.warmup, rounds=args.rounds,
                         repeat=args.repeat, seed=args.seed),
                winner_knobs={f"B{b}_S{s}": knobs for (b, s), knobs in WINNERS.items()})
    dump_json(out / "run_info.json", info)
    print(f"GPU={info['gpu_name']} Git={git_sha[:12]} llvm={info['llvm_flavor']}")

    rows = []
    resolved = {}
    # Fail on numerical errors; raw timings are LOCAL ONLY.
    for b, s in selected:
        shape = f"B{b}_S{s}"
        print(f"\n--- {shape} ---", flush=True)
        req = AttentionRequest(batch=b, nhead_q=32, nhead_k=8,
                               seqlen_q=s, seqlen_k=s, hdim_q=128, hdim_v=128,
                               arch="gfx942", mask_type=1, dtype="bf16",
                               algorithm="attention_dense", spec_id="gfx942_dense")
        baseline = attention_tuning_spec(req, "gfx942_dense").kernel_spec
        winner = replace(baseline, **WINNERS[(b, s)])
        specs = {"production": baseline, "winner": winner}
        if winner.causal_diag_split:
            print("  STACKED: SKIPPED — winner has causal_diag_split=True, incompatible with BF16 CFVST")
            incompatible = "winner has causal_diag_split=True; changing it would no longer be the winning configuration"
        else:
            specs["winner_plus_full"] = replace(winner, use_cfvst=True,
                                                lds_num_buffers=2,
                                                v_reg_prefetch=True)
            incompatible = None
        names = {}
        for arm, spec in specs.items():
            ok, reason = dense.supports_attention_dense(spec, arch="gfx942")
            if not ok:
                if arm == "winner_plus_full":
                    incompatible = reason
                    print(f"  STACKED: SKIPPED — kernel rejects exact winner + full: {reason}")
                    del specs[arm]
                    continue
                raise RuntimeError(f"{shape}/{arm} unsupported: {reason}")
            names[arm] = dense.gfx942_kernel_name(spec)
        if len(set(names.values())) != len(names):
            raise RuntimeError(f"{shape}: kernel-name collision: {names}")
        resolved[shape] = {
            "winner_overrides": WINNERS[(b, s)],
            "stacked_incompatible_reason": incompatible,
            "arms": {arm: {
                "spec": asdict(spec), "kernel": names[arm],
                "persistent": spec.persistent,
                "exp2_fast": spec.resolved_use_exp2_fast(),
                "lds_bytes": dense._lds_bytes(spec),
            } for arm, spec in specs.items()},
        }
        dump_json(out / "resolved_specs.json", resolved)

        torch.manual_seed(args.seed)
        q = (torch.randn((b, s, 32, 128), device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        k = (torch.randn((b, s, 8, 128), device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        v = (torch.randn_like(k) * 0.2).contiguous()
        scale = 1.0 / math.sqrt(128)
        reference = fp32_reference(torch, q, k, v)
        prepared = {}
        for arm, spec in specs.items():
            output = torch.empty_like(q)
            art = compile_kernel(dense.build_attention_dense(spec, arch="gfx942"),
                                 arch="gfx942", backend="python", capture_ir_text=False)
            if art.kernel_name != names[arm]:
                raise RuntimeError(f"{arm}: compiled symbol differs from spec kernel name")
            launch = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name,
                                    signature=dense.attention_dense_signature(spec))
            config = LaunchConfig(grid=dense.attention_dense_grid(spec),
                                  block=dense.attention_dense_block(spec), stream=stream_id)
            values = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": output,
                      "scale": scale, **dense.attention_dense_runtime_args(spec)}
            output.fill_(float("nan"))
            launch(values, config=config)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output.float(), reference, atol=1e-2, rtol=1e-2)
            err = float((output.float() - reference).abs().max().item())
            print(f"  PASS {arm}: max_abs={err:.6g}, kernel={names[arm]}", flush=True)
            prepared[arm] = (launch, values, config)
        del reference
        for _ in range(args.warmup):
            for arm in prepared:
                l, vls, cfg = prepared[arm]
                l(vls, config=cfg)
        torch.cuda.synchronize(device)
        samples = {arm: [] for arm in prepared}
        for round_index in range(args.rounds):
            order = list(prepared)
            rng.shuffle(order)
            for arm in order:
                l, vls, cfg = prepared[arm]
                ms = timed(torch, l, vls, cfg, stream, args.repeat)
                samples[arm].append(ms)
                print(f"    round {round_index+1}: {arm:18s} {ms:.6f} ms")
        med = {arm: statistics.median(values) for arm, values in samples.items()}
        base_ms = med["production"]
        winner_ms = med["winner"]
        for arm in prepared:
            ms = med[arm]
            ratio = base_ms / ms
            stacked_ratio = winner_ms / ms if arm == "winner_plus_full" else None
            rows.append({"shape": shape, "arm": arm, "median_ms": f"{ms:.9f}",
                         "speedup_vs_production": f"{ratio:.6f}",
                         "speedup_vs_winner": "" if stacked_ratio is None else f"{stacked_ratio:.6f}",
                         "min_ms": f"{min(samples[arm]):.9f}",
                         "max_ms": f"{max(samples[arm]):.9f}"})
            print(f"  {arm:18s} {ratio:.5f}x vs production"
                  + (f", {stacked_ratio:.5f}x vs winner" if stacked_ratio else ""))
        if incompatible:
            rows.append({"shape": shape, "arm": "winner_plus_full", "median_ms": "",
                         "speedup_vs_production": "", "speedup_vs_winner": "",
                         "min_ms": "", "max_ms": ""})
        with (out / "local_raw_timings.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Public-safe ratios-only report, but review provenance before publishing.
    lines = ["# gfx942 BF16 D128 causal — production, winner, winner + optimizations",
             "", "Local draft; six pre-selected sweep winners, no new sweep performed.",
             "3 variants per shape except incompatible winners; all measured variants passed FP32 checks.",
             "", "| B | S | Winner / production | Winner+full / production | Winner+full / winner |",
             "|---:|---:|---:|---:|---:|"]
    lookup = {(r["shape"], r["arm"]): r for r in rows}
    for b, s in selected:
        key = f"B{b}_S{s}"
        w = lookup.get((key, "winner"))
        o = lookup.get((key, "winner_plus_full"))
        lines.append(f"| {b} | {s} | {w['speedup_vs_production']}x | "
                     f"{o['speedup_vs_production']}x | {o['speedup_vs_winner']}x |"
                     if o and o["speedup_vs_production"] else
                     f"| {b} | {s} | {w['speedup_vs_production']}x | incompatible | incompatible |")
    lines += ["", "For B=1, S=1024, winning `causal_diag_split=True` cannot be stacked with BF16 CFVST.",
              "These results reflect only the specified GPU, software and six shapes; no general superiority claim.", ""]
    (out / "three_run_ratios.md").write_text("\n".join(lines))
    print(f"\nCompleted. Report: {out / 'three_run_ratios.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())