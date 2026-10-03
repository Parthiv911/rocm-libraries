#!/usr/bin/env python3
"""Faithful same-process 5-way gfx942 attention_dense ablation.

The accompanying ``attention_dense_ablation.py`` is a LOCAL-ONLY emitter copy.
It contains four compile-time scheduling bodies:

  baseline           : CFVST off, historical single-buffer loop
  cfvst              : CFVST on, actual historical pre-KDB/VPF single-buffer loop
  cfvst_klds         : CFVST + K-LDS double buffering / early K only
  cfvst_vpf          : CFVST + next-V VGPR prefetch only
  cfvst_klds_vpf     : CFVST + KDB + VPF (current full pipeline)

All five kernels are compiled once in ONE Python process.  Timing calls the already
constructed ``KernelLauncher`` objects directly (no run_attention_dense_torch/cache
lookup inside the timed region), matching the user's older ROCKE benchmark method.
All timing uses the same torch/HIP stream and randomized interleaved rounds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _find_rocke_root(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if (p / "library/kernels/gfx942/attention_dense.py").is_file():
            return p
        raise SystemExit(f"--rocke-root is not a ROCKE root: {p}")
    if os.environ.get("ROCKE_ROOT"):
        p = Path(os.environ["ROCKE_ROOT"]).expanduser().resolve()
        if (p / "library/kernels/gfx942/attention_dense.py").is_file():
            return p
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in (start, *start.parents):
            if (p / "library/kernels/gfx942/attention_dense.py").is_file():
                return p
            nested = p / "dnn-providers/hip-kernel-provider/rocke"
            if (nested / "library/kernels/gfx942/attention_dense.py").is_file():
                return nested
    raise SystemExit("Could not locate ROCKE; pass --rocke-root or set ROCKE_ROOT")


def _install_paths(rocke: Path) -> None:
    for p in (rocke / "library", rocke / "platform/Python", rocke):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_ablation_emitter():
    path = Path(__file__).resolve().with_name("attention_dense_ablation.py")
    if not path.is_file():
        raise SystemExit(f"missing benchmark emitter copy: {path}")
    name = "rocke_gfx942_attention_dense_ablation_local"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Variant:
    name: str
    cfvst: bool
    mode: str
    kdb: bool
    vpf: bool


VARIANTS = (
    Variant("baseline", False, "legacy", False, False),
    Variant("cfvst", True, "legacy", False, False),
    Variant("cfvst_klds", True, "klds", True, False),
    Variant("cfvst_vpf", True, "vpf", False, True),
    Variant("cfvst_klds_vpf", True, "full", True, True),
)


@dataclass(frozen=True)
class Shape:
    batch: int
    seqlen: int
    hq: int
    hkv: int
    d: int
    dtype: str

    @property
    def label(self) -> str:
        return f"b{self.batch}_s{self.seqlen}_hq{self.hq}_hkv{self.hkv}_d{self.d}_{self.dtype}"


def _parse_shape(text: str) -> Shape:
    xs = [x.strip() for x in text.split(",")]
    if len(xs) != 6:
        raise argparse.ArgumentTypeError("shape: B,S,HQ,HKV,D,dtype")
    try:
        b, s, hq, hkv, d = map(int, xs[:5])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    dtype = xs[5].lower()
    if dtype not in ("bf16", "fp16"):
        raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")
    if d != 128:
        raise argparse.ArgumentTypeError("this CFVST/KDB/VPF ablation is D128-only")
    if hq <= 0 or hkv <= 0 or hq % hkv:
        raise argparse.ArgumentTypeError("HQ must be divisible by HKV")
    return Shape(b, s, hq, hkv, d, dtype)


def _torch_dtype(torch, dtype: str):
    return torch.bfloat16 if dtype == "bf16" else torch.float16


def _make_reference(torch, q, k, v, *, scale: float, causal: bool, chunk: int):
    bsz, sq, hq, d = q.shape
    _, sk, hkv, _ = k.shape
    gqa = hq // hkv
    qf = q.float().permute(0, 2, 1, 3).contiguous()
    kf = k.float().permute(0, 2, 1, 3).contiguous()
    vf = v.float().permute(0, 2, 1, 3).contiguous()
    out = torch.empty((bsz, hq, sq, d), device=q.device, dtype=torch.float32)
    key_idx = torch.arange(sk, device=q.device).view(1, 1, 1, sk)
    with torch.no_grad():
        for hk in range(hkv):
            hs, he = hk * gqa, (hk + 1) * gqa
            kt = kf[:, hk : hk + 1].transpose(-1, -2)
            vv = vf[:, hk : hk + 1]
            for qs in range(0, sq, chunk):
                qe = min(qs + chunk, sq)
                scores = torch.matmul(qf[:, hs:he, qs:qe], kt) * float(scale)
                if causal:
                    qi = torch.arange(qs, qe, device=q.device).view(1, 1, -1, 1)
                    scores.masked_fill_(key_idx > qi, float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[:, hs:he, qs:qe] = torch.matmul(probs, vv)
    return out.permute(0, 2, 1, 3).contiguous()


def _validate(torch, got, ref, *, rtol: float, atol: float) -> tuple[float, float]:
    gotf = got.float()
    diff = (gotf - ref).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / ref.abs().clamp_min(1.0e-6)).max().item())
    torch.testing.assert_close(gotf, ref, rtol=rtol, atol=atol)
    return max_abs, max_rel


def _time_launcher(torch, launcher, vals, cfg, stream, repeat: int) -> float:
    # IMPORTANT: only direct launcher calls are inside the event interval.
    # No compile/cache lookup/spec resolution or run_attention_dense_torch wrapper.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeat):
        launcher(vals, config=cfg)
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) / repeat


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocke-root")
    ap.add_argument("--shape", action="append", type=_parse_shape)
    ap.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--persistent", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--num-persistent", type=int, default=304)
    ap.add_argument("--persist-decode", choices=("auto", "qb_major", "hkv_major"), default="auto")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--seed", type=int, default=942)
    ap.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reference-chunk", type=int, default=128)
    ap.add_argument("--rtol-bf16", type=float, default=3.0e-2)
    ap.add_argument("--atol-bf16", type=float, default=3.0e-2)
    ap.add_argument("--rtol-fp16", type=float, default=1.0e-2)
    ap.add_argument("--atol-fp16", type=float, default=1.0e-2)
    ap.add_argument("--out-dir", default="results/5way")
    args = ap.parse_args()
    if args.rounds < 3:
        raise SystemExit("--rounds must be >= 3")
    if args.repeat <= 0 or args.warmup < 0:
        raise SystemExit("repeat > 0 and warmup >= 0 required")

    rocke = _find_rocke_root(args.rocke_root)
    _install_paths(rocke)

    import torch
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    dense = _load_ablation_emitter()
    if not torch.cuda.is_available():
        raise SystemExit("No HIP/CUDA device visible to torch")

    shapes = args.shape or [Shape(1, 4096, 32, 8, 128, "bf16")]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    stream = torch.cuda.current_stream(device=device)
    stream_ptr = int(stream.cuda_stream)
    rng = random.Random(args.seed)

    raw_rows, summary_rows, validation_rows = [], [], []

    print(f"ROCKE_ROOT={rocke}")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"stream_ptr={stream_ptr}")
    print(f"variants={[v.name for v in VARIANTS]}")

    for shape in shapes:
        if shape.seqlen % 256 or shape.seqlen % 64:
            raise SystemExit(f"{shape.label}: S must be divisible by block_m=256 and block_n=64")
        work = (shape.seqlen // 256) * shape.hq * shape.batch
        persistent = (
            True if args.persistent == "on" else
            False if args.persistent == "off" else
            work >= args.num_persistent
        )
        base_kwargs = dict(
            batch=shape.batch,
            seqlen_q=shape.seqlen,
            seqlen_kv=shape.seqlen,
            num_query_heads=shape.hq,
            num_kv_heads=shape.hkv,
            head_size=shape.d,
            causal=args.causal,
            dtype=shape.dtype,
            block_m=256,
            block_n=64,
            persistent=persistent,
            num_persistent=args.num_persistent,
            persist_decode=args.persist_decode,
            waves_per_eu=dense._tuned_waves_per_eu(shape.d, shape.dtype),
        )

        specs = {}
        for v in VARIANTS:
            sp = dense.Gfx942AttentionDenseSpec(
                **base_kwargs,
                use_cfvst=v.cfvst,
                bench_pipeline_mode=v.mode,
            )
            ok, why = dense.supports_attention_dense(sp, arch="gfx942")
            if not ok:
                raise SystemExit(f"{shape.label}/{v.name}: unsupported: {why}")
            specs[v.name] = sp

        names = {n: sp.kernel_name() for n, sp in specs.items()}
        if len(set(names.values())) != 5:
            raise RuntimeError(f"kernel-name collision: {names}")

        tdtype = _torch_dtype(torch, shape.dtype)
        torch.manual_seed(0)
        q = (torch.randn((shape.batch, shape.seqlen, shape.hq, shape.d), device=device, dtype=tdtype) * 0.2).contiguous()
        k = (torch.randn((shape.batch, shape.seqlen, shape.hkv, shape.d), device=device, dtype=tdtype) * 0.2).contiguous()
        vv = (torch.randn_like(k) * 0.2).contiguous()
        outs = {v.name: torch.empty_like(q) for v in VARIANTS}
        scale = 1.0 / math.sqrt(shape.d)

        print(f"\n[{shape.label}] persistent={persistent} np={args.num_persistent}")
        for v in VARIANTS:
            sp = specs[v.name]
            print(
                f"  {v.name:20s} mode={v.mode:6s} cfvst={int(v.cfvst)} "
                f"kbufs={dense._k_nbuf(sp)} lds={dense._lds_bytes(sp)}B\n"
                f"    {names[v.name]}"
            )

        # Compile every binary before timing, then create a direct KernelLauncher.
        launchers, cfgs, vals = {}, {}, {}
        print("  compiling all five variants...")
        for v in VARIANTS:
            sp = specs[v.name]
            art = compile_kernel(
                dense.build_attention_dense(sp, arch="gfx942"),
                arch="gfx942",
                backend="python",
                capture_ir_text=False,
            )
            if art.kernel_name != sp.kernel_name():
                raise RuntimeError(f"artifact/name mismatch for {v.name}: {art.kernel_name} != {sp.kernel_name()}")
            launchers[v.name] = KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=dense.attention_dense_signature(sp),
            )
            cfgs[v.name] = LaunchConfig(
                grid=dense.attention_dense_grid(sp),
                block=dense.attention_dense_block(sp),
                stream=stream_ptr,
            )
            vals[v.name] = {
                "q_ptr": q,
                "k_ptr": k,
                "v_ptr": vv,
                "o_ptr": outs[v.name],
                "scale": scale,
            }

        # Correctness first, via the exact same direct launchers used for timing.
        if args.validate:
            print("  building chunked fp32 reference...")
            ref = _make_reference(torch, q, k, vv, scale=scale, causal=args.causal, chunk=args.reference_chunk)
            rtol = args.rtol_bf16 if shape.dtype == "bf16" else args.rtol_fp16
            atol = args.atol_bf16 if shape.dtype == "bf16" else args.atol_fp16
            for v in VARIANTS:
                launchers[v.name](vals[v.name], config=cfgs[v.name])
                torch.cuda.synchronize(device)
                ma, mr = _validate(torch, outs[v.name], ref, rtol=rtol, atol=atol)
                validation_rows.append(dict(shape=shape.label, variant=v.name, passed=1, max_abs=ma, max_rel=mr, rtol=rtol, atol=atol))
                print(f"  validate {v.name:20s} PASS max_abs={ma:.6g} max_rel={mr:.6g}")
            del ref

        # Round-robin warmup so no single variant gets the cold-device bias.
        for _ in range(args.warmup):
            for v in VARIANTS:
                launchers[v.name](vals[v.name], config=cfgs[v.name])
        torch.cuda.synchronize(device)

        # Load-bearing timing: same process, same stream, randomized/interleaved rounds.
        samples = {v.name: [] for v in VARIANTS}
        for r in range(1, args.rounds + 1):
            order = list(VARIANTS)
            rng.shuffle(order)
            print(f"  round {r}: {[v.name for v in order]}")
            for pos, v in enumerate(order, 1):
                ms = _time_launcher(torch, launchers[v.name], vals[v.name], cfgs[v.name], stream, args.repeat)
                samples[v.name].append(ms)
                raw_rows.append(dict(
                    shape=shape.label, round=r, order=pos, variant=v.name,
                    mode=v.mode, cfvst=int(v.cfvst), kdb=int(v.kdb), vpf=int(v.vpf),
                    repeat=args.repeat, ms=f"{ms:.9f}"
                ))
                print(f"    {v.name:20s} {ms:.6f} ms")

        med = {name: statistics.median(xs) for name, xs in samples.items()}
        base, cf, full = med["baseline"], med["cfvst"], med["cfvst_klds_vpf"]
        for v in VARIANTS:
            m = med[v.name]
            summary_rows.append(dict(
                shape=shape.label, variant=v.name, mode=v.mode,
                median_ms=f"{m:.9f}", speedup_vs_baseline=f"{base/m:.6f}",
                speedup_vs_cfvst=f"{cf/m:.6f}", min_ms=f"{min(samples[v.name]):.9f}",
                max_ms=f"{max(samples[v.name]):.9f}", rounds=args.rounds, repeat=args.repeat,
            ))

        print("  medians:")
        for v in VARIANTS:
            print(f"    {v.name:20s} {med[v.name]:.6f} ms")
        print("  same-session ratios:")
        print(f"    CFVST                 baseline/cfvst = {base/cf:.4f}x")
        print(f"    KDB over CFVST      cfvst/cfvst_klds = {cf/med['cfvst_klds']:.4f}x")
        print(f"    VPF over CFVST        cfvst/cfvst_vpf = {cf/med['cfvst_vpf']:.4f}x")
        print(f"    VPF after KDB   cfvst_klds/full = {med['cfvst_klds']/full:.4f}x")
        print(f"    KDB after VPF     cfvst_vpf/full = {med['cfvst_vpf']/full:.4f}x")
        print(f"    Total                  baseline/full = {base/full:.4f}x")

    _write_rows(out_dir / "ablation_runs.csv",
                ["shape","round","order","variant","mode","cfvst","kdb","vpf","repeat","ms"], raw_rows)
    _write_rows(out_dir / "ablation_summary.csv",
                ["shape","variant","mode","median_ms","speedup_vs_baseline","speedup_vs_cfvst","min_ms","max_ms","rounds","repeat"], summary_rows)
    if args.validate:
        _write_rows(out_dir / "validation.csv",
                    ["shape","variant","passed","max_abs","max_rel","rtol","atol"], validation_rows)
    (out_dir / "measurement_conditions.txt").write_text("\n".join([
        f"rocke_root={rocke}", f"device={torch.cuda.get_device_name(device)}",
        f"torch={torch.__version__}", f"hip={getattr(torch.version, 'hip', None)}",
        f"stream_ptr={stream_ptr}", f"warmup={args.warmup}", f"rounds={args.rounds}",
        f"repeat={args.repeat}", f"seed={args.seed}", f"validate={args.validate}",
        f"causal={args.causal}", f"persistent={args.persistent}",
        f"num_persistent={args.num_persistent}", f"persist_decode={args.persist_decode}",
        "timed_call=direct KernelLauncher (no run_attention_dense_torch wrapper)",
        "cfvst_reference=historical pre-KDB/VPF single-buffer loop",
    ]) + "\n")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())