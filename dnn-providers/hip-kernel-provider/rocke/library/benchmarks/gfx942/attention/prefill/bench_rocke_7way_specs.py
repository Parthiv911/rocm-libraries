#!/usr/bin/env python3
"""Eight-arm persistent gfx942 BF16 D128 attention: Step-0 winner A vs seven-stage B ablation.

A is the exact shape-specific existing-lever sweep winner from the supplied
12-row table (no CFVST/KDB/VPF). B[1..7] always start from the CURRENT
per-shape production-dispatch baseline, independently of A. Thus this is a
fair new same-source A/B measurement, NOT a claim about cross-commit timings.

   A sweep_winner        per-shape best preexisting levers (new pipeline off)
   1 baseline            production dispatch (CFVST/KDB/VPF off)
   2 cfvst               baseline + CFVST
   3 cfvst_kdb           CFVST + K LDS double buffering
   4 cfvst_vpf           CFVST + V register prefetch (KDB off)
   5 pipeline            CFVST + KDB + V register prefetch
   6 pipeline_pad        full pipeline + lds_row_pad=4
   7 pipeline_pad_dsplit previous + causal_diag_split=True

Stages 3 and 4 are separate branches from CFVST alone. Stage 5 combines
both branches. A is the ONLY arm receiving lever-sweep geometry, scheduling,
and grid knobs. The seven B configurations do not inherit A's knobs.
The exact single-knob B deltas are checked against production dispatch.
The original expected A kernel names are retained and checked with warnings
for symbol-name differences; resolved specs, HSACO hashes and artifacts are kept.

Methodology: torch-free NumPy/rocKE, one process and explicit HIP stream,
shared Q/K/V and chunked NumPy FP32 causal GQA reference over BF16 inputs,
BF16 correctness before timing and after every timed batch,
round-robin warmup, five randomized rounds, 50 launches per arm per round,
median latency and same-round paired min/max speedups.
provenance/source hashes are recorded.
Absolute latency results are kept in local CSVs.
Default cohort: the eight persistent-304 BF16 D128 causal shapes with A winners.
The four nonpersistent S=1024/2048 shapes are excluded because the support
contract gates KDB/VPF to persistent=True. Custom --shape values must have an
explicitly supplied Step-0 winner AND select persistent production dispatch.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import random
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass

B_VARIANTS = (
    "baseline",
    "cfvst",
    "cfvst_kdb",
    "cfvst_vpf",
    "pipeline",
    "pipeline_pad",
    "pipeline_pad_dsplit",
)
VARIANTS = ("A", *B_VARIANTS)
STAGE_NUMBERS = {"A": 0, **{name: i for i, name in enumerate(B_VARIANTS, start=1)}}

STAGE_DESCRIPTIONS = {
    "A": "Per-shape best existing-lever sweep winner (CFVST/KDB/VPF off)",
    "baseline": "Production-dispatch baseline, without CFVST/KDB/VPF",
    "cfvst": "Baseline + conflict-free V store transpose",
    "cfvst_kdb": "CFVST + K LDS double buffering (no V prefetch)",
    "cfvst_vpf": "CFVST + V register prefetch (no K double buffering)",
    "pipeline": "CFVST + K LDS double buffering + V register prefetch",
    "pipeline_pad": "Full pipeline + lds_row_pad=4",
    "pipeline_pad_dsplit": "Full pipeline + lds_row_pad=4 + causal_diag_split=True",
}

# Dependency graph (not a linear chain): both KDB and VPF extend CFVST.
# The full pipeline is cross-checked against both upstream branches.
PARENT_VARIANT = {
    "A": None,
    "baseline": None,
    "cfvst": "baseline",
    "cfvst_kdb": "cfvst",
    "cfvst_vpf": "cfvst",
    "pipeline": "cfvst_kdb",
    "pipeline_pad": "pipeline",
    "pipeline_pad_dsplit": "pipeline_pad",
}

STAGE_OVERRIDES = {
    "A": {},  # Separate sweep-winner specification; not a B stage.
    "baseline": {},
    "cfvst": {"use_cfvst": True},
    "cfvst_kdb": {"use_cfvst": True, "lds_num_buffers": 2},
    "cfvst_vpf": {"use_cfvst": True, "v_reg_prefetch": True},
    "pipeline": {"use_cfvst": True, "lds_num_buffers": 2, "v_reg_prefetch": True},
    "pipeline_pad": {
        "use_cfvst": True,
        "lds_num_buffers": 2,
        "v_reg_prefetch": True,
        "lds_row_pad": 4,
    },
    "pipeline_pad_dsplit": {
        "use_cfvst": True,
        "lds_num_buffers": 2,
        "v_reg_prefetch": True,
        "lds_row_pad": 4,
        "causal_diag_split": True,
    },
}


# Step-0 winners from the supplied 12-row final A/B table. These values
# are not applied to any B stage. Expected symbols are used for audit only.
# Keys: (batch, seqlen, hkv). Hq=32, D=128, BF16, causal for every row.

# ONLY previously swept levers go into A. Unspecified levers stay at their

# production-dispatch values; the new CFVST/KDB/VPF pipeline stays OFF.

# All entries come from the user-supplied twelve-row winner table.

WINNING_LEVERS = {
    (1, 1024, 8): dict(
        block_m=128,
        block_n=32,
        waves_per_eu=4,
        lds_row_pad=4,
        use_exp2_fast=True,
        causal_diag_split=True,
    ),
    (1, 1024, 16): dict(
        block_m=128,
        block_n=32,
        waves_per_eu=4,
        lds_row_pad=4,
        use_exp2_fast=True,
        causal_diag_split=True,
    ),
    (1, 2048, 8): dict(
        block_m=128,
        block_n=32,
        waves_per_eu=4,
        lds_row_pad=4,
        use_exp2_fast=True,
        pv_loop_order="k_major",
        causal_diag_split=True,
    ),
    (1, 2048, 16): dict(
        block_m=128,
        block_n=32,
        waves_per_eu=4,
        lds_row_pad=4,
        use_exp2_fast=True,
        pv_loop_order="k_major",
        causal_diag_split=True,
    ),
    (1, 4096, 8): dict(
        waves_per_eu=3,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=1,
        causal_diag_split=True,
    ),
    (1, 4096, 16): dict(
        waves_per_eu=2,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=2,
        causal_diag_split=True,
    ),
    (1, 8192, 8): dict(
        waves_per_eu=2,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=2,
        causal_diag_split=True,
    ),
    (1, 8192, 16): dict(
        waves_per_eu=3,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=3,
        causal_diag_split=True,
    ),
    (1, 16384, 8): dict(
        waves_per_eu=2,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=1,
        causal_diag_split=True,
    ),
    (16, 4096, 8): dict(
        waves_per_eu=1,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=3,
        causal_diag_split=True,
    ),
    (16, 4096, 16): dict(
        waves_per_eu=1,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=2,
        causal_diag_split=True,
    ),
    (16, 8192, 8): dict(
        waves_per_eu=3,
        lds_row_pad=4,
        iglp=True,
        iglp_mode=1,
        pv_loop_order="k_major",
        pv_priority=1,
        causal_diag_split=True,
    ),
}


# Auditing only: current source may rename a kernel without changing its knobs.

# The script warns (and records both symbols) rather than rejecting a new HEAD.

EXPECTED_A_KERNEL = {
    (
        1,
        1024,
        8,
    ): "rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit",
    (
        1,
        1024,
        16,
    ): "rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_dsplit",
    (
        1,
        2048,
        8,
    ): "rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit",
    (
        1,
        2048,
        16,
    ): "rocke_attention_dense_d128_bn32_bf16_bm128_causal_lazyrs_gfx942_wpe4_krowpad4_e2f1_pvkmaj_dsplit",
    (
        1,
        4096,
        8,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit",
    (
        1,
        4096,
        16,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit",
    (
        1,
        8192,
        8,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit",
    (
        1,
        8192,
        16,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit",
    (
        1,
        16384,
        8,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe2_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit",
    (
        16,
        4096,
        8,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio3_dsplit",
    (
        16,
        4096,
        16,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_gfx942_wpe1_krowpad4_iglp1_iglpm1_pvkmaj_prio2_dsplit",
    (
        16,
        8192,
        8,
    ): "rocke_attention_dense_d128_bn64_bf16_causal_lazyrs_persist304_hkvmaj_gfx942_wpe3_krowpad4_iglp1_iglpm1_pvkmaj_prio1_dsplit",
}


# Anchor A to the original sweep's NON-new-pipeline values rather than

# inheriting potentially changed policies in the current dispatch.

A_COMMON = dict(
    block_m=256,
    block_n=64,
    lds_row_pad=4,
    lds_k_group_pad=8,
    use_exp2_fast=True,
    iglp=False,
    iglp_mode=0,
    pv_loop_order="d_major",
    pv_priority=0,
    pv_sched_fence_mask=None,
    o_store_width=4,
    causal_diag_split=True,
    interleave=False,
    lazy_rescale=True,
    emit_lse=False,
    use_cfvst=False,
    lds_num_buffers=1,
    v_reg_prefetch=False,
)


def original_grid_overrides(shape: "Shape") -> dict:
    """Pin A's original default/304-CTA and qb-/hkv-major layouts."""

    if shape.batch == 1 and shape.seqlen <= 2048:

        return dict(
            persistent=False,
            num_persistent=304,
            persist_decode="auto",
            nonpersist_decode="qb_minor",
        )

    persist_decode = "hkv_major" if shape.batch == 16 and shape.hkv == 8 else "qb_major"

    return dict(
        persistent=True,
        num_persistent=304,
        persist_decode=persist_decode,
        nonpersist_decode="auto",
    )


# speedup = median(numerator)/median(denominator)
# Paired min/max = extrema of same-round numerator/denominator ratios.
# Each effect is measured relative to its correct parent, not list position.
COMPARISON_PAIRS = (
    ("baseline", "cfvst", "cfvst_only"),
    ("cfvst", "cfvst_kdb", "kdb_given_cfvst"),
    ("cfvst", "cfvst_vpf", "vpf_given_cfvst"),
    ("cfvst_kdb", "pipeline", "vpf_given_cfvst_kdb"),
    ("cfvst_vpf", "pipeline", "kdb_given_cfvst_vpf"),
    ("baseline", "cfvst_kdb", "cfvst_plus_kdb_total"),
    ("baseline", "cfvst_vpf", "cfvst_plus_vpf_total"),
    ("baseline", "pipeline", "pipeline_total"),
    ("pipeline", "pipeline_pad", "lds_pad"),
    ("pipeline_pad", "pipeline_pad_dsplit", "causal_split"),
    ("baseline", "pipeline_pad_dsplit", "total"),
    # A is the best earlier Step-0 lever winner; each B stage uses a separate
    # production-derived spec. A/B results are remeasured on the SAME checkout.
    *(("A", name, f"A_vs_{name}") for name in B_VARIANTS),
)
EFFECT_NAMES = tuple(effect for _, _, effect in COMPARISON_PAIRS)


def changed_fields(before, after) -> set[str]:
    """Confirm cumulative stages differ only in their advertised knobs."""
    b, a = asdict(before), asdict(after)
    return {key for key in b if b[key] != a[key]}


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
        return (
            f"b{self.batch}_s{self.seqlen}_hq{self.hq}_hkv{self.hkv}"
            f"_d{self.d}_{self.dtype}"
        )


# Persistent-only default cohort. Exclude B=1 S=1024/2048, where dispatch is
# nonpersistent and supports_attention_dense rejects all KDB/VPF stages.
# Compile, validate and time all seven B arms plus A on each eligible shape.


DEFAULT_SHAPES = (
    Shape(1, 4096, 32, 8, 128, "bf16"),
    Shape(1, 4096, 32, 16, 128, "bf16"),
    Shape(1, 8192, 32, 8, 128, "bf16"),
    Shape(1, 8192, 32, 16, 128, "bf16"),
    Shape(1, 16384, 32, 8, 128, "bf16"),
    Shape(16, 4096, 32, 8, 128, "bf16"),
    Shape(16, 4096, 32, 16, 128, "bf16"),
    Shape(16, 8192, 32, 8, 128, "bf16"),
)


def parse_shape(value: str) -> Shape:
    try:
        parts = [s.strip() for s in value.split(",")]
        if len(parts) != 6:
            raise ValueError("expected exactly six comma-separated entries")
        b, s, hq, hkv, d = (int(x) for x in parts[:5])
        dtype = parts[5].lower()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--shape must be B,S,Hq,Hkv,D,dtype, e.g. 1,4096,32,8,128,bf16"
        ) from exc
    # The pipeline supports BF16 D128.
    if dtype != "bf16" or d != 128:
        raise argparse.ArgumentTypeError("seven-way ablation requires bf16 D128")
    if min(b, s, hq, hkv) <= 0 or hq % hkv:
        raise argparse.ArgumentTypeError(
            "positive extents and Hq divisible by Hkv required"
        )
    if s % 256 or s % 64:
        raise argparse.ArgumentTypeError(
            "S must be divisible by block_m=256 and block_n=64"
        )
    return Shape(b, s, hq, hkv, d, dtype)


def locate_rocke(explicit: str | None) -> Path:
    starts = []
    if explicit:
        starts.append(Path(explicit).expanduser())
    starts.extend((Path.cwd(), Path(__file__).resolve().parent))
    for start in starts:
        for path in (start, *start.parents):
            for candidate in (path, path / "dnn-providers/hip-kernel-provider/rocke"):
                if (candidate / "library/kernels/gfx942/attention_dense.py").is_file():
                    return candidate.resolve()
    raise SystemExit("ROCKE root not found: pass --rocke-root")


def add_module_paths(rocke: Path) -> None:
    for p in (rocke / "library", rocke / "platform/python", rocke):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def fp32_reference(q, k, v, *, scale: float, chunk: int):
    """Chunked causal GQA SDPA in NumPy FP32 over the exact BF16 inputs.

    Batches and KV heads are processed separately. Peak score matrix is
    (Hq/Hkv, chunk, S) so largest shapes do not allocate an S-by-S tensor.
    """
    batch, sq, hq, d = q.shape
    _, sk, hkv, _ = k.shape
    gqa = hq // hkv
    result = np.empty((batch, sq, hq, d), dtype=np.float32)
    key_indices = np.arange(sk)[None, :]
    for b in range(batch):
        for h in range(hkv):
            first, last = h * gqa, (h + 1) * gqa
            kt = k[b, :, h, :].T
            val = v[b, :, h, :]
            for qs in range(0, sq, chunk):
                qe = min(qs + chunk, sq)
                query = q[b, qs:qe, first:last, :].transpose(1, 0, 2)
                scores = np.matmul(query, kt)
                scores *= np.float32(scale)
                # Bottom-left aligned causal mask: key_index <= query_index.
                masked = key_indices > np.arange(qs, qe)[:, None]
                np.copyto(scores, -np.inf, where=masked[None, :, :])
                scores -= np.max(scores, axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= np.sum(scores, axis=-1, keepdims=True)
                result[b, qs:qe, first:last, :] = np.matmul(scores, val).transpose(
                    1, 0, 2
                )
    return result


def validate(actual_bits, reference, *, atol: float, rtol: float, chunk: int = 256):
    """Compare every BF16 output element to FP32 reference, chunked on host."""
    from rocke.numeric.bf16 import bf16_to_f32

    if actual_bits.shape != reference.shape:
        raise AssertionError(
            f"output shape {actual_bits.shape} != reference {reference.shape}"
        )
    if not np.isfinite(reference).all():
        raise AssertionError("FP32 reference contains nonfinite values")
    max_abs = 0.0
    max_rel = 0.0
    for b in range(reference.shape[0]):
        for qs in range(0, reference.shape[1], chunk):
            sl = slice(qs, qs + chunk)
            actual32 = bf16_to_f32(actual_bits[b, sl])
            expected = reference[b, sl]
            np.testing.assert_allclose(
                actual32, expected, atol=atol, rtol=rtol, equal_nan=False
            )
            diff = np.abs(actual32 - expected)
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(
                max_rel, float((diff / np.maximum(np.abs(expected), 1e-6)).max())
            )
    return max_abs, max_rel


def _device(array):
    """Adam's torch-free ROCm transfer: NumPy -> rocKE DeviceMem."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem

    array = np.ascontiguousarray(array)
    mem = DeviceMem(array.nbytes)
    Runtime().memcpy_h2d(mem.ptr(), as_u8_buffer(array), array.nbytes)
    return mem


def _download_bf16(device_buffer, shape):
    """Read raw BF16 uint16 bits using rocKE's HIP host-copy utility."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer

    bits = np.empty(shape, dtype=np.uint16)
    Runtime().memcpy_d2h(as_u8_buffer(bits), device_buffer.ptr(), bits.nbytes)
    return bits


@contextmanager
def _hip_stream():
    """A dedicated HIP stream, avoiding framework-dependent stream lookup."""
    from rocke.runtime import hip_module
    from rocke.runtime.hip_module import Runtime

    hip = hip_module._resolve_hip()  # Same HIP DSO used by rocKE.
    create = hip.hipStreamCreate
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int
    destroy = hip.hipStreamDestroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    hip_module._check(create(ctypes.byref(handle)), "hipStreamCreate")
    if not handle.value:
        raise RuntimeError("hipStreamCreate returned null HIP stream")
    stream = int(handle.value)
    try:
        yield stream
    finally:
        try:
            Runtime().wait_stream(stream)
        finally:
            hip_module._check(destroy(handle), "hipStreamDestroy")


def timed_batch(launcher, values, config, stream: int, repeat: int) -> float:
    """HIP events around exactly N launches; preserves old launch fencing.

    Equivalent to the original torch.cuda.Event harness: no changed
    LaunchConfig fence setting, no compilation or correctness checks timed.
    """
    from rocke.runtime.hip_module import Runtime

    first = Runtime().event()
    last = Runtime().event()
    try:
        first.record(stream=stream)
        for _ in range(repeat):
            launcher(values, config=config)
        last.record(stream=stream)
        last.synchronize()
        return float(first.elapsed_to(last)) / repeat
    finally:
        first.destroy()
        last.destroy()


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _command_output(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=12
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_provenance(root: Path) -> dict:
    repo = _command_output(["git", "rev-parse", "--show-toplevel"], cwd=root)
    head = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _command_output(["git", "branch", "--show-current"], cwd=root)
    status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    )
    return {
        "repository_root": repo,
        "head_sha": head,
        "branch": branch,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _capture_git_checkout(root: Path) -> dict:
    """Record Git provenance; do NOT block dirty/untracked/unknown checkout."""
    state = _git_provenance(root)
    print(
        f"Git HEAD={state['head_sha'] or 'unknown'}; "
        f"dirty={state['working_tree_dirty']} (allowed)",
        flush=True,
    )
    return state


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _loaded_hip_runtime_info() -> dict:
    """Identify the HIP runtime actually used by rocKE, without loading another.

    rocKE caches the ctypes handle in hip_module._resolve_hip(). Resolve the
    version symbol on THAT handle. Linux /proc/self/maps identifies the file
    backing the symbol's executable mapping, not a second libamdhip64 found
    via LD_LIBRARY_PATH or an arbitrary package version.
    """
    from rocke.runtime import hip_module

    hip = hip_module._resolve_hip()
    version_fn = hip.hipRuntimeGetVersion
    version_fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    version_fn.restype = ctypes.c_int
    encoded_version = ctypes.c_int(-1)
    status = version_fn(ctypes.byref(encoded_version))
    if status != 0 or encoded_version.value <= 0:
        raise RuntimeError(
            "hipRuntimeGetVersion on rocKE's loaded HIP library failed: "
            f"status={status}, version={encoded_version.value}"
        )

    symbol_ptr = ctypes.cast(version_fn, ctypes.c_void_p).value
    if not symbol_ptr:
        raise RuntimeError("Cannot resolve hipRuntimeGetVersion function address")

    # Resolve the actual mapped shared object from the symbol's instruction
    # address. /proc/self/maps is available on the Linux ROCm benchmark host.
    mapped_path = None
    try:
        with Path("/proc/self/maps").open(encoding="utf-8") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                lo, hi = (int(x, 16) for x in fields[0].split("-", 1))
                if lo <= symbol_ptr < hi:
                    if fields[5].startswith("/"):
                        mapped_path = fields[5]
                    break
    except OSError as exc:
        raise RuntimeError("Cannot inspect /proc/self/maps for HIP library") from exc
    if mapped_path is None:
        raise RuntimeError(
            "Cannot identify the mapped HIP shared object containing "
            "hipRuntimeGetVersion; refusing ambiguous runtime provenance"
        )

    # HIP_VERSION encoding, not CUDA's encoding:
    # major*10_000_000 + minor*100_000 + patch.
    raw = encoded_version.value
    major = raw // 10_000_000
    minor = (raw // 100_000) % 100
    patch = raw % 100_000
    return {
        "version": f"{major}.{minor}.{patch}",
        "version_encoded": raw,
        "loaded_library_path": mapped_path,
        "rocke_library_opened_as": str(hip._name),
        "query": "hipRuntimeGetVersion from rocke.runtime.hip_module's cached HIP handle",
    }


# These are ROCm components that can be loaded by rocKE or the
# compiler. The package version is NOT necessarily the version of every library
# loaded into the process, so capture the actual mapped shared objects too.
_ROCM_SHARED_OBJECT = re.compile(
    r"^lib(?:amdhip64|hsa-runtime64|amd_comgr|hiprtc|hipblas(?:lt)?|"
    r"rocblas|rocsolver|rocsparse|rocfft|hipfft|hiprand|rocrand|"
    r"miopen|rccl|roctracer64|rocprofiler(?:-sdk)?(?:64)?|"
    r"rocprim|hipcub)(?:[.-]|$)",
    flags=re.IGNORECASE,
)


def _rocm_sdk_package_versions() -> dict[str, str]:
    """Installed ROCm wheel metadata, not a claim about loaded DSOs."""
    distributions = (
        "rocm",
        "rocm-sdk",
        "rocm-sdk-core",
        "rocm-sdk-runtime",
        "rocm-sdk-libraries",
        "rocm-hip-runtime",
        "rocm-comgr",
        "_rocm_sdk_core",
    )
    return {
        name: version
        for name in distributions
        if (version := _package_version(name)) is not None
    }


def _loaded_rocm_shared_objects() -> list[dict]:
    """Actual ROCm shared objects mapped into THIS running Linux process.

    Read /proc/self/maps to get mapped file identity instead of searching PATH,
    LD_LIBRARY_PATH, or an installation directory. The same basename can be
    loaded from different directories, so preserve every distinct mapping.
    This does not infer an unexposed per-library ROCm release version.
    """
    found: dict[tuple[str, str, str], dict] = {}
    try:
        with Path("/proc/self/maps").open(encoding="utf-8") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                mapped_path = fields[5]
                if not mapped_path.startswith("/"):
                    continue
                # A mapped library might have been unlinked after loading.
                name = Path(mapped_path.removesuffix(" (deleted)")).name
                if not _ROCM_SHARED_OBJECT.match(name):
                    continue
                key = (mapped_path, fields[3], fields[4])
                found[key] = {
                    "name": name,
                    "mapped_path": mapped_path,
                    "device": fields[3],
                    "inode": fields[4],
                }
    except OSError as exc:
        raise RuntimeError("Cannot inspect /proc/self/maps for ROCm libraries") from exc
    return sorted(found.values(), key=lambda r: (r["name"], r["mapped_path"]))


def _environment(
    device_id: int, rocke: Path, script_file: Path, git_state: dict
) -> dict:
    # Hardware identity comes from rocKE's HIP runtime, not PyTorch properties.
    from rocke.core.lower_llvm import _resolve_llvm_flavor
    from rocke.runtime.comgr import loaded_compiler_info
    from rocke.runtime.hip_module import (
        get_device_name,
        get_device_num_cus,
        get_device_target_id,
    )

    gpu = {
        "name": get_device_name(device_id),
        "architecture": get_device_target_id(device_id),
        "cu_count_reported_by_hip": get_device_num_cus(device_id),
    }
    # Record the loaded COMGR compiler rather than an environment override or
    # an unrelated clang executable found on PATH.
    compiler = loaded_compiler_info()
    llvm_flavor = _resolve_llvm_flavor()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "software": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "hip_runtime": _loaded_hip_runtime_info(),
            "triton": _package_version("triton"),
            "rocm_sdk_package_versions": _rocm_sdk_package_versions(),
        },
        "compile_backend": "python",
        "llvm_flavor": llvm_flavor,
        "loaded_compiler": {
            "llvm_version": (
                ".".join(map(str, compiler.llvm_version))
                if compiler is not None and compiler.llvm_version is not None
                else None
            ),
            "detection_source": compiler.source if compiler is not None else None,
            "comgr_library": compiler.comgr_path if compiler is not None else None,
            "query_library": (
                compiler.query_library_path if compiler is not None else None
            ),
        },
        "git": git_state,
        "source_files": {
            "attention_dense.py": str(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "attention_dense_sha256": _file_sha256(
                rocke / "library/kernels/gfx942/attention_dense.py"
            ),
            "benchmark_script": str(script_file),
            "benchmark_script_sha256": _file_sha256(script_file),
            "builder_sha256": _file_sha256(
                rocke
                / "library/builders/gfx942/attention/prefill/attention_dense_prefill.py"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rocke-root", help="Directory containing library/kernels/gfx942"
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="B,S,Hq,Hkv,128,bf16 (repeatable; overrides default eight persistent shapes)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--reference-chunk", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out-dir", default="~/rocke_sevenway_ablation_results")
    args = parser.parse_args()
    if args.rounds < 3 or args.repeat <= 0 or args.warmup < 0:
        parser.error("require rounds >= 3, repeat > 0 and warmup >= 0")
    if args.reference_chunk <= 0:
        parser.error("reference-chunk must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol and rtol must be nonnegative")

    rocke = locate_rocke(args.rocke_root)
    measured_git = _capture_git_checkout(rocke)
    add_module_paths(rocke)

    import kernels.gfx942.attention_dense as dense
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig
    from rocke.runtime.hip_module import (
        current_device,
        get_device_arch,
        get_device_count,
    )
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    if get_device_count() == 0:
        raise SystemExit("ROCm GPU is not visible through HIP")
    for required in ("Gfx942AttentionDenseSpec", "attention_dense_runtime_args"):
        if not hasattr(dense, required):
            raise SystemExit(f"Expected updated attention_dense.py; missing {required}")
    source_file = Path(dense.__file__).resolve()
    expected_file = (rocke / "library/kernels/gfx942/attention_dense.py").resolve()
    if source_file != expected_file:
        raise SystemExit(
            f"Loaded wrong emitter! expected {expected_file}, got {source_file}"
        )
    needed_knobs = {
        knob for overrides in STAGE_OVERRIDES.values() for knob in overrides
    }
    needed_knobs.update(A_COMMON)
    needed_knobs.update(
        {"persistent", "num_persistent", "persist_decode", "nonpersist_decode"}
    )
    for overrides in WINNING_LEVERS.values():
        needed_knobs.update(overrides)
    for knob in sorted(needed_knobs):
        if knob not in dense.Gfx942AttentionDenseSpec.__dataclass_fields__:
            raise SystemExit(f"Current Gfx942AttentionDenseSpec is missing knob {knob}")
    rng = random.Random(args.seed)
    shapes = args.shape if args.shape else list(DEFAULT_SHAPES)
    # A's exact winning configurations are defined only for the supplied cohort.
    for shape in shapes:
        key = (shape.batch, shape.seqlen, shape.hkv)
        if shape.hq != 32 or key not in WINNING_LEVERS:
            parser.error(
                f"No supplied Step-0 winning kernel for {shape.label}; "
                "choose one of the twelve table shapes"
            )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device_id = current_device()
    if get_device_arch(device_id) != "gfx942":
        raise SystemExit("This benchmark requires a gfx942 GPU")
    environment = _environment(device_id, rocke, Path(__file__).resolve(), measured_git)
    hip_runtime = environment["software"]["hip_runtime"]
    print(
        f"ROCKE_ROOT={rocke} | GPU={environment['gpu']['name']} "
        f"({environment['gpu']['cu_count_reported_by_hip']} CUs) | "
        f"Git={environment['git']['head_sha']} "
        f"(dirty={environment['git']['working_tree_dirty']}) | "
        f"LLVM={environment['llvm_flavor']}"
    )
    print(
        f"HIP runtime={hip_runtime['version']} | DSO={hip_runtime['loaded_library_path']}"
    )
    # Stream is released even if compilation/numerical validation fails.
    with _hip_stream() as stream_ptr:
        return _run_shapes(args, rocke, environment, shapes, out_dir, rng, stream_ptr)


def _run_shapes(args, rocke, environment, shapes, out_dir, rng, stream_ptr) -> int:
    import kernels.gfx942.attention_dense as dense
    from dispatch.attention import AttentionRequest, attention_tuning_spec
    from rocke.helpers.compile import compile_kernel
    from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
    from rocke.runtime import KernelLauncher, LaunchConfig
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem

    rt = Runtime()
    print(f"HIP stream={stream_ptr} | stages={list(VARIANTS)}")

    raw_rows: list[dict] = []
    validation_rows: list[dict] = []
    summary_rows: list[dict] = []
    ratio_rows: list[dict] = []
    comparison_rows: list[dict] = []
    resolved_rows: list[dict] = []
    print(f"Running {len(shapes)} shape(s): {[s.label for s in shapes]}")
    for shape_index, shape in enumerate(shapes, start=1):
        print(f"\n=== Shape {shape_index}/{len(shapes)}: {shape.label} ===", flush=True)
        # Exact production request: mirror the gfx942 dense builder's dense_request
        # (BF16 causal, square self-attention), rather than hand-building a spec.
        req = AttentionRequest(
            batch=shape.batch,
            nhead_q=shape.hq,
            nhead_k=shape.hkv,
            seqlen_q=shape.seqlen,
            seqlen_k=shape.seqlen,
            hdim_q=shape.d,
            hdim_v=shape.d,
            arch="gfx942",
            mask_type=1,  # causal; same as dense_request(..., causal=True)
            dtype=shape.dtype,
            algorithm="attention_dense",
            spec_id="gfx942_dense",
        )
        production = attention_tuning_spec(req, "gfx942_dense").kernel_spec
        if not isinstance(production, dense.Gfx942AttentionDenseSpec):
            raise RuntimeError(
                f"{shape.label}: dispatch returned {type(production).__name__}, "
                "not Gfx942AttentionDenseSpec"
            )
        if not production.persistent:
            raise RuntimeError(
                f"{shape.label}: production dispatch selected persistent=False; "
                "KDB/VPF are gated to persistent=True. This eight-arm benchmark "
                "requires a persistent shape."
            )

        # A = exact shape-specific best EXISTING-LEVER kernel, independently
        # compiled on the current emitter. B1-B7 = current production + only
        # the explicit new-pipeline/pad/split overrides (no sweep-lever overlay).
        key = (shape.batch, shape.seqlen, shape.hkv)
        a_overrides = A_COMMON | original_grid_overrides(shape) | WINNING_LEVERS[key]
        specs = {"A": replace(production, **a_overrides), "baseline": production}
        for name in B_VARIANTS[1:]:
            specs[name] = replace(production, **STAGE_OVERRIDES[name])

        original_a_name = EXPECTED_A_KERNEL[key]
        actual_a_name = specs["A"].kernel_name()
        if actual_a_name != original_a_name:
            print(
                "  WARNING: rebuilt A symbol differs from original sweep; "
                "check current source/spec differences.\n"
                f"    original={original_a_name}\n"
                f"    current ={actual_a_name}",
                flush=True,
            )
        if (
            specs["A"].resolved_use_cfvst()
            or specs["A"].lds_num_buffers != 1
            or specs["A"].v_reg_prefetch
        ):
            raise RuntimeError(f"{shape.label}: A unexpectedly enables CFVST/KDB/VPF")

        if (
            production.resolved_use_cfvst()
            or production.lds_num_buffers != 1
            or production.v_reg_prefetch
        ):
            raise RuntimeError(
                f"{shape.label}: production dispatch already enables some "
                "CFVST/KDB/VPF pipeline knobs. Cannot label it a clean baseline."
            )
        expected_steps = (
            ("baseline", "cfvst", {"use_cfvst"}),
            ("cfvst", "cfvst_kdb", {"lds_num_buffers"}),
            ("cfvst", "cfvst_vpf", {"v_reg_prefetch"}),
            ("cfvst_kdb", "pipeline", {"v_reg_prefetch"}),
            ("cfvst_vpf", "pipeline", {"lds_num_buffers"}),
            ("pipeline", "pipeline_pad", {"lds_row_pad"}),
            ("pipeline_pad", "pipeline_pad_dsplit", {"causal_diag_split"}),
        )
        for before, after, expected in expected_steps:
            actual = changed_fields(specs[before], specs[after])
            if actual != expected:
                raise RuntimeError(
                    f"{shape.label}: {before} -> {after} changed {sorted(actual)}, "
                    f"expected {sorted(expected)}. Current dispatch may already "
                    "enable a lever; refusing a misleading ablation."
                )
        # Check resolved policy too: default/None fields might silently resolve
        # differently than their literal dataclass values.
        for name in B_VARIANTS:
            spec = specs[name]
            expected = STAGE_OVERRIDES[name]
            observed_pipeline = (
                bool(spec.resolved_use_cfvst()),
                int(spec.lds_num_buffers),
                bool(spec.v_reg_prefetch),
            )
            expected_pipeline = (
                bool(expected.get("use_cfvst", False)),
                int(expected.get("lds_num_buffers", 1)),
                bool(expected.get("v_reg_prefetch", False)),
            )
            if observed_pipeline != expected_pipeline:
                raise RuntimeError(
                    f"{shape.label}/{name}: resolved pipeline knobs "
                    f"{observed_pipeline} differ from expected {expected_pipeline}"
                )
        for name, spec in specs.items():
            ok, reason = dense.supports_attention_dense(spec, arch="gfx942")
            if not ok:
                raise RuntimeError(f"{shape.label}/{name} rejected: {reason}")

        # Trace full specs, including policy resolutions that dataclass fields
        # otherwise encode as None.
        def describe_spec(spec):
            return {
                "fields": asdict(spec),
                "resolved_use_cfvst": spec.resolved_use_cfvst(),
                "resolved_use_exp2_fast": spec.resolved_use_exp2_fast(),
                "resolved_persist_decode": spec.resolved_persist_decode,
                "resolved_v_row_pad": spec.resolved_v_row_pad(),
                "resolved_use_v_swizzle": spec.resolved_use_v_swizzle(),
                "kernel_name": spec.kernel_name(),
                "grid": dense.attention_dense_grid(spec),
                "block": dense.attention_dense_block(spec),
            }

        resolved_rows.append(
            {
                "shape": shape.label,
                "request": asdict(req),
                "source": "dispatch.attention.attention_tuning_spec(req, 'gfx942_dense').kernel_spec",
                "production_baseline": True,
                "production_dispatch_kernel": production.kernel_name(),
                "A_role": "per-shape Step-0 lever-sweep winner, independently rebuilt",
                "A_original_expected_kernel": original_a_name,
                "A_current_kernel": actual_a_name,
                "A_kernel_name_matches_original": actual_a_name == original_a_name,
                "A_overrides": a_overrides,
                "B_stage_baseline": "current production dispatch",
                "stage_descriptions": STAGE_DESCRIPTIONS,
                "stage_overrides": STAGE_OVERRIDES,
                "parent_variant": PARENT_VARIANT,
                "comparison_pairs": COMPARISON_PAIRS,
                "arms": {name: describe_spec(spec) for name, spec in specs.items()},
            }
        )
        for name, spec in specs.items():
            print(f"  {STAGE_NUMBERS[name]} {name:20s}: {spec.kernel_name()}")
        print(
            "  A source: Step-0 winner; B stages: current production dispatch "
            "with pipeline-only overrides"
        )
        print(
            f"  dispatch: persistent={production.persistent}, "
            f"NP={production.num_persistent}, "
            f"decode={production.resolved_persist_decode}, "
            f"exp2_fast={production.resolved_use_exp2_fast()}, "
            f"bm/bn={production.block_m}/{production.block_n}, "
            f"wpe={production.waves_per_eu}"
        )
        names = {name: spec.kernel_name() for name, spec in specs.items()}
        if len(set(names.values())) != len(VARIANTS):
            raise RuntimeError(f"Compile-time kernel-name collision: {names}")

        # Match the original two BF16-rounding points: random BF16, multiply
        # by 0.2, round back to BF16. The NumPy RNG differs from torch's RNG.
        input_rng = np.random.default_rng(0)
        q_shape = (shape.batch, shape.seqlen, shape.hq, 128)
        kv_shape = (shape.batch, shape.seqlen, shape.hkv, 128)

        def bf16_random_scaled(dims):
            original = input_rng.standard_normal(dims, dtype=np.float32)
            initial_bf16 = bf16_to_f32(f32_to_bf16(original))
            return f32_to_bf16(initial_bf16 * np.float32(0.2))

        q_bits = bf16_random_scaled(q_shape)
        k_bits = bf16_random_scaled(kv_shape)
        v_bits = bf16_random_scaled(kv_shape)
        q = _device(q_bits)
        k = _device(k_bits)
        v_tensor = _device(v_bits)
        outputs = {name: DeviceMem(q_bits.nbytes) for name in VARIANTS}
        scale = 1.0 / math.sqrt(128)
        launchers = {}
        configurations = {}
        values = {}
        compiled_artifacts = {}
        print(
            f"  Compiling {len(VARIANTS)} variants independently and saving HSACOs..."
        )
        for name in VARIANTS:
            spec = specs[name]
            artifact = compile_kernel(
                dense.build_attention_dense(spec, arch="gfx942"),
                arch="gfx942",
                backend="python",
                capture_ir_text=False,
            )
            hsaco_dir = out_dir / "hsaco" / shape.label
            hsaco_dir.mkdir(parents=True, exist_ok=True)

            hsaco_path = hsaco_dir / f"{name}.hsaco"
            hsaco_path.write_bytes(artifact.hsaco)
            if artifact.kernel_name != names[name]:
                raise RuntimeError(f"Artifact/kernel name mismatch: {name}")
            compiled_artifacts[name] = {
                "kernel_name": artifact.kernel_name,
                "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
                "lds_bytes": dense._lds_bytes(spec),
            }
            launchers[name] = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=dense.attention_dense_signature(spec),
            )
            configurations[name] = LaunchConfig(
                grid=dense.attention_dense_grid(spec),
                block=dense.attention_dense_block(spec),
                stream=stream_ptr,
            )
            values[name] = {
                "q_ptr": q,
                "k_ptr": k,
                "v_ptr": v_tensor,
                "o_ptr": outputs[name],
                "scale": scale,
                # REQUIRED for updated runtime-shape kernel ABI and fast division.
                **dense.attention_dense_runtime_args(spec),
            }

        resolved_rows[-1]["compiled_artifacts"] = compiled_artifacts
        print("Building one chunked FP32 NumPy SDPA reference...")
        reference = fp32_reference(
            bf16_to_f32(q_bits),
            bf16_to_f32(k_bits),
            bf16_to_f32(v_bits),
            scale=scale,
            chunk=args.reference_chunk,
        )
        del q_bits, k_bits, v_bits

        def check_and_record(name: str, stage: str, round_id: int | str = ""):
            """Validate against the same FP32 reference and record the exact gate."""
            try:
                bits = _download_bf16(outputs[name], reference.shape)
                max_abs, max_rel = validate(
                    bits, reference, atol=args.atol, rtol=args.rtol
                )
            except (AssertionError, RuntimeError) as exc:
                raise RuntimeError(
                    f"ABORT: correctness failed: {shape.label}/{name}, "
                    f"stage={stage}, round={round_id}"
                ) from exc
            validation_rows.append(
                dict(
                    shape=shape.label,
                    variant=name,
                    stage=stage,
                    round=round_id,
                    passed=1,
                    max_abs=f"{max_abs:.9g}",
                    max_rel=f"{max_rel:.9g}",
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            return max_abs, max_rel

        print(f"  Validating all {len(VARIANTS)} compiled kernels...")
        for name in VARIANTS:
            # Validate through exactly the launcher/config used for timed calls.
            # NaN poison catches elements not written by a kernel.
            rt.memset(outputs[name].ptr(), 0xFF, outputs[name].nbytes())
            rt.sync()  # memset runs on default HIP stream
            launchers[name](values[name], config=configurations[name])
            rt.wait_stream(stream_ptr)
            max_abs, max_rel = check_and_record(name, "pre_timing")
            print(f"  PASS {name:20s} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        # Keep the FP32 reference alive for post-batch validation.
        rt.wait_stream(stream_ptr)

        print(f"Round-robin warmup: {args.warmup} launches per variant")
        for _ in range(args.warmup):
            for name in VARIANTS:
                launchers[name](values[name], config=configurations[name])
        rt.wait_stream(stream_ptr)

        samples = {name: [] for name in VARIANTS}
        for round_id in range(1, args.rounds + 1):
            shuffled = list(VARIANTS)
            rng.shuffle(shuffled)
            print(f"Round {round_id}: {shuffled}")
            for position, name in enumerate(shuffled, 1):
                ms = timed_batch(
                    launchers[name],
                    values[name],
                    configurations[name],
                    stream_ptr,
                    args.repeat,
                )
                # Timed events have already synchronized. Only the last output
                # of this batch survives; validate it outside the timed window.
                max_abs, max_rel = check_and_record(name, "post_batch", round_id)
                samples[name].append(ms)
                raw_rows.append(
                    dict(
                        shape=shape.label,
                        round=round_id,
                        order=position,
                        variant=name,
                        use_cfvst=int(specs[name].resolved_use_cfvst()),
                        lds_num_buffers=specs[name].lds_num_buffers,
                        v_reg_prefetch=int(specs[name].v_reg_prefetch),
                        lds_row_pad=specs[name].lds_row_pad,
                        causal_diag_split=int(specs[name].causal_diag_split),
                        repeat=args.repeat,
                        ms=f"{ms:.9f}",
                    )
                )
                print(f"  {name:20s} {ms:.6f} ms | PASS post-batch (abs={max_abs:.4g})")

        del reference
        rt.wait_stream(stream_ptr)
        median = {name: statistics.median(times) for name, times in samples.items()}
        print(
            "  Median ms: " + ", ".join(f"{name}={median[name]:.6f}" for name in median)
        )
        for name in VARIANTS:
            x = median[name]
            summary_rows.append(
                dict(
                    shape=shape.label,
                    variant=name,
                    kernel_name=names[name],
                    median_ms=f"{x:.9f}",
                    min_ms=f"{min(samples[name]):.9f}",
                    max_ms=f"{max(samples[name]):.9f}",
                    speedup_vs_baseline=f"{median['baseline']/x:.9f}",
                    parent_variant=PARENT_VARIANT[name] or "",
                    speedup_vs_parent=(
                        f"{median[PARENT_VARIANT[name]]/x:.9f}"
                        if PARENT_VARIANT[name] is not None
                        else "1.000000000"
                    ),
                    stage=STAGE_NUMBERS[name],
                    description=STAGE_DESCRIPTIONS[name],
                    rounds=args.rounds,
                    repeat=args.repeat,
                )
            )

        if any(x <= 0 for x in median.values()):
            raise RuntimeError("GPU timing returned nonpositive latency")
        pair_metrics = {}
        for numerator, denominator, effect in COMPARISON_PAIRS:
            before, after = median[numerator], median[denominator]
            paired = [x / y for x, y in zip(samples[numerator], samples[denominator])]
            speedup = before / after
            reduction = 100.0 * (before - after) / before
            metrics = dict(
                shape=shape.label,
                effect=effect,
                numerator=numerator,
                denominator=denominator,
                speedup=f"{speedup:.9f}",
                latency_reduction_pct=f"{reduction:.6f}",
                paired_min=f"{min(paired):.9f}",
                paired_max=f"{max(paired):.9f}",
            )
            ratio_rows.append(metrics)
            pair_metrics[effect] = metrics
        comparison_rows.append(
            {
                "batch": shape.batch,
                "seqlen": shape.seqlen,
                "hkv": shape.hkv,
                **{f"{name}_kernel": names[name] for name in VARIANTS},
                **{f"{name}_median_ms": f"{median[name]:.9f}" for name in VARIANTS},
                **{
                    f"{effect}_speedup": pair_metrics[effect]["speedup"]
                    for effect in EFFECT_NAMES
                },
                **{
                    f"{effect}_reduction_pct": pair_metrics[effect][
                        "latency_reduction_pct"
                    ]
                    for effect in EFFECT_NAMES
                },
            }
        )
        print("  Incremental, branch-specific and cumulative results:", flush=True)
        for effect in EFFECT_NAMES:
            r = pair_metrics[effect]
            print(
                f"    {effect:28s} {float(r['speedup']):.5f}x "
                f"({float(r['latency_reduction_pct']):+.2f}% latency reduction; "
                f"paired {float(r['paired_min']):.5f}-{float(r['paired_max']):.5f}x)",
                flush=True,
            )
        rt.wait_stream(stream_ptr)
        del outputs, values, q, k, v_tensor, launchers, configurations

    (out_dir / "resolved_specs.json").write_text(
        json.dumps(resolved_rows, indent=2, default=str) + "\n"
    )
    # Self-contained eight-arm schemas; no external analyzer assumptions.
    exports = (
        (
            "ablation_comparison.csv",
            "batch seqlen hkv "
            + " ".join(f"{name}_kernel" for name in VARIANTS)
            + " "
            + " ".join(f"{name}_median_ms" for name in VARIANTS)
            + " "
            + " ".join(f"{effect}_speedup" for effect in EFFECT_NAMES)
            + " "
            + " ".join(f"{effect}_reduction_pct" for effect in EFFECT_NAMES),
            comparison_rows,
        ),
        (
            "ablation_runs.csv",
            "shape round order variant use_cfvst lds_num_buffers v_reg_prefetch "
            "lds_row_pad causal_diag_split repeat ms",
            raw_rows,
        ),
        (
            "ablation_summary.csv",
            "shape variant kernel_name median_ms min_ms max_ms "
            "speedup_vs_baseline parent_variant speedup_vs_parent stage description rounds repeat",
            summary_rows,
        ),
        (
            "validation.csv",
            "shape variant stage round passed max_abs max_rel atol rtol",
            validation_rows,
        ),
        (
            "ablation_ratios.csv",
            "shape effect numerator denominator speedup latency_reduction_pct paired_min paired_max",
            ratio_rows,
        ),
    )
    for filename, schema, rows in exports:
        write_csv(out_dir / filename, tuple(schema.split()), rows)
    # Speedup-only Markdown; raw absolute timings remain in local CSV files.
    ratios_by_shape = {}
    for r in ratio_rows:
        ratios_by_shape.setdefault(r["shape"], {})[r["effect"]] = r

    def fmt(r, effect):
        x = r[effect]
        return (
            f"{float(x['speedup']):.5f}x "
            f"[{float(x['paired_min']):.5f}, "
            f"{float(x['paired_max']):.5f}]"
        )

    lines = [
        "# A (Step-0 winner) versus B1-B7 BF16 D128 gfx942 ablation",
        "",
        "## Variant definitions",
        "A is the exact shape-specific existing-lever sweep winner, rebuilt on the current emitter. ",
        "B1-B7 are derived separately from the CURRENT production dispatch. ",
        "A's optimization knobs are **not** added to any B stage. ",
        "Original A symbols and current compiled A symbols are recorded in resolved_specs.json.",
        "",
        "| # | Variant | Knobs enabled | Parent comparison |",
        "|---:|---|---|---|",
    ]
    for name in VARIANTS:
        overrides = STAGE_OVERRIDES[name]
        knobs = (
            "Shape-specific existing-lever winner (see resolved_specs.json)"
            if name == "A"
            else ", ".join(f"`{k}={v}`" for k, v in overrides.items())
            or "Production dispatch"
        )
        parent = PARENT_VARIANT[name] or "—"
        lines.append(f"| {STAGE_NUMBERS[name]} | `{name}` | {knobs} | `{parent}` |")
    lines += [
        "",
        "## A versus the production-derived B stages (new same-source A/B)",
        "| B | S | Hkv | A / B1 baseline | A / B2 CFVST | A / B3 CFVST+KDB | A / B4 CFVST+VPF | A / B5 full | A / B6 full+pad | A / B7 full+pad+split |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape[sh.label]
        effects = tuple(f"A_vs_{name}" for name in B_VARIANTS)
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    lines += [
        "",
        "## Direct B ablations (production baseline / CFVST anchor)",
        "| B | S | Hkv | CFVST / baseline | CFVST+KDB / CFVST | CFVST+VPF / CFVST | Full / CFVST+KDB | Full / CFVST+VPF |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape[sh.label]
        effects = (
            "cfvst_only",
            "kdb_given_cfvst",
            "vpf_given_cfvst",
            "vpf_given_cfvst_kdb",
            "kdb_given_cfvst_vpf",
        )
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    lines += [
        "",
        "## Cumulative, padding and causal-split ablation",
        "| B | S | Hkv | Full / baseline | + LDS pad (6/5) | + Causal split (7/6) | Total (7/1) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sh in shapes:
        r = ratios_by_shape[sh.label]
        effects = ("pipeline_total", "lds_pad", "causal_split", "total")
        lines.append(
            f"| {sh.batch} | {sh.seqlen} | {sh.hkv} | "
            + " | ".join(fmt(r, effect) for effect in effects)
            + " |"
        )
    lines += [
        "",
        "Speedup = numerator-arm median latency divided by denominator-arm median "
        "latency. Bracketed range = min/max of same-round paired latency ratios. "
        "A value >1x means the denominator arm is faster.",
        "",
        "## Compiled kernels",
        "| B | S | Hkv | "
        + " | ".join(f"{STAGE_NUMBERS[name]}: {name}" for name in VARIANTS)
        + " |",
        "|---:|---:|---:|" + "---|" * len(VARIANTS),
    ]
    for r in comparison_rows:
        lines.append(
            f"| {r['batch']} | {r['seqlen']} | {r['hkv']} | "
            + " | ".join(f"`{r[f'{name}_kernel']}`" for name in VARIANTS)
            + " |"
        )
    (out_dir / "ablation_comparison.md").write_text("\n".join(lines) + "\n")
    # A-focused compact report retains the expected symbol, current symbol,
    # paired A/B min/max ratios and medians without mixing old sweep timings.
    a_vs_b_rows = []
    for sh in shapes:
        key = (sh.batch, sh.seqlen, sh.hkv)
        r = next(
            row
            for row in comparison_rows
            if (row["batch"], row["seqlen"], row["hkv"]) == key
        )
        p = ratios_by_shape[sh.label]
        a_vs_b_rows.append(
            {
                "batch": sh.batch,
                "seqlen": sh.seqlen,
                "hkv": sh.hkv,
                "A_original_kernel": EXPECTED_A_KERNEL[key],
                "A_compiled_kernel": r["A_kernel"],
                "A_kernel_name_matches_original": r["A_kernel"]
                == EXPECTED_A_KERNEL[key],
                "B7_kernel": r["pipeline_pad_dsplit_kernel"],
                "A_median_ms": r["A_median_ms"],
                "B7_median_ms": r["pipeline_pad_dsplit_median_ms"],
                "speedup_A_over_B7": p["A_vs_pipeline_pad_dsplit"]["speedup"],
                "latency_reduction_pct": p["A_vs_pipeline_pad_dsplit"][
                    "latency_reduction_pct"
                ],
                "paired_min": p["A_vs_pipeline_pad_dsplit"]["paired_min"],
                "paired_max": p["A_vs_pipeline_pad_dsplit"]["paired_max"],
            }
        )
    write_csv(out_dir / "A_vs_B7.csv", tuple(a_vs_b_rows[0]), a_vs_b_rows)
    environment["run"] = {
        "shapes": [asdict(sh) for sh in shapes],
        "variant_count": len(VARIANTS),
        "correctness_atol": args.atol,
        "correctness_rtol": args.rtol,
        "reference": "chunked NumPy FP32 causal GQA SDPA over BF16-rounded inputs",
        "input_dtype": "bf16",
        "warmup_per_arm": args.warmup,
        "rounds": args.rounds,
        "launches_per_round_per_arm": args.repeat,
        "randomization_seed": args.seed,
        "input_seed": 0,
        "input_rng": "numpy.random.default_rng(0)",
        "input_encoding": "NumPy FP32, BF16 RNE, multiply 0.2, BF16 RNE",
        "hip_stream_handle": stream_ptr,
        "timing": "rocKE Runtime HIP events around 50 direct KernelLauncher calls, unchanged launcher fencing",
        "validation_scope": "pre-timing and final output after every timed batch",
        "validation_checks_per_shape": len(VARIANTS) * (1 + args.rounds),
        "all_validation_passed": all(v["passed"] == 1 for v in validation_rows),
        "dispatch_baseline_for_all_shapes": True,
        "A_is_per_shape_step0_winner": True,
        "A_expected_kernels": {str(k): v for k, v in EXPECTED_A_KERNEL.items()},
        "B_stages_derive_only_from_production": True,
        "stage_descriptions": STAGE_DESCRIPTIONS,
        "stage_overrides": STAGE_OVERRIDES,
        "parent_variant": PARENT_VARIANT,
        "comparison_pairs": COMPARISON_PAIRS,
        "dirty_checkout_allowed": True,
        "exact_python_command": shlex.join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
    }
    # Snapshot after compilation/launch so lazily loaded ROCm DSOs appear.
    environment["software"][
        "rocm_libraries_loaded_at_end"
    ] = _loaded_rocm_shared_objects()
    (out_dir / "provenance.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )
    print(f"\nOutputs in {out_dir}")
    print(
        "Reproducibility: provenance.json, resolved_specs.json, validation.csv, "
        "ablation_ratios.csv, ablation_comparison.csv, ablation_comparison.md, A_vs_B7.csv; "
        "keep latency CSVs internal"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
