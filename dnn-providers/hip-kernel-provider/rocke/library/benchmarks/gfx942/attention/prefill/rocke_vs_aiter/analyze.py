python - <<'PY'
import json
from pathlib import Path

ROOT = Path("sweep_results")
README = Path("README.md")

START = "<!-- BEGIN MEASURED CONFIGURATIONS -->"
END = "<!-- END MEASURED CONFIGURATIONS -->"

KNOBS = (
    "use_cfvst", "lds_row_pad", "lds_k_group_pad",
    "persistent", "block_m", "block_n", "num_persistent",
    "persist_decode", "interleave", "use_v_swizzle",
    "v_row_pad", "use_exp2_fast", "iglp", "iglp_mode",
    "pv_sched_fence_mask", "pv_priority", "pv_loop_order",
    "o_store_width", "causal_diag_split", "waves_per_eu",
    "lds_num_buffers", "lazy_rescale", "emit_lse",
)

assert len(KNOBS) == 23

def full_knobs(spec):
    missing = set(KNOBS) - set(spec)
    if missing:
        raise ValueError(f"Missing knob values: {sorted(missing)}")
    return json.dumps(
        {k: spec[k] for k in sorted(KNOBS)},
        indent=2
    )

results = []

for b in (1, 16):
    for s in (1024, 2048, 4096):
        directory = ROOT / f"B{b}_S{s}_Hq32_Hkv8_D128"

        baseline = json.loads(
            (directory / "baseline.json").read_text()
        )
        summary = json.loads(
            (directory / "final_validation.json").read_text()
        )["summary"]

        winner = summary["best_nonbaseline"]
        winner_spec = summary["best_nonbaseline_full_spec"]

        if not winner or not winner_spec:
            raise ValueError(f"Missing winner: B={b}, S={s}")

        if baseline["kernel_name"] != summary["baseline_kernel"]:
            raise ValueError(f"Baseline mismatch: B={b}, S={s}")

        for k, v in winner["knobs"].items():
            if winner_spec[k] != v:
                raise ValueError(f"Winner mismatch: {k}")

        full_knobs(baseline["spec"])
        full_knobs(winner_spec)

        results.append((b, s, baseline, winner, winner_spec))

lines = [
    "| B | S | Speedup | Paired min–max | Baseline kernel | Winning kernel |",
    "|---:|---:|---:|---|---|---|",
]

for b, s, baseline, winner, spec in results:
    lines.append(
        f'| {b} | {s} | **{winner["ratio"]:.5f}×** | '
        f'{winner["min"]:.5f}–{winner["max"]:.5f}× | '
        f'`{baseline["kernel_name"]}` | '
        f'`{winner["kernel"]}` |'
    )

for b, s, baseline, winner, spec in results:
    lines.extend([
        "",
        f"### B={b}, S={s}",
        "",
        f'**Baseline kernel:** `{baseline["kernel_name"]}`',
        "",
        "**Baseline knobs:**",
        "",
        "```json",
        full_knobs(baseline["spec"]),
        "```",
        "",
        f'**Winning kernel:** `{winner["kernel"]}`',
        "",
        "**Winning knobs:**",
        "",
        "```json",
        full_knobs(spec),
        "```",
    ])

section = "\n".join(lines)

original = README.read_text() if README.exists() else ""

if START in original and END in original:
    before = original.split(START, 1)[0]
    after = original.split(END, 1)[1]
    updated = before + START + "\n\n" + section + "\n\n" + END + after
else:
    updated = (
        original.rstrip()
        + "\n\n## Baseline vs winning configurations\n\n"
        + START + "\n\n"
        + section + "\n\n"
        + END + "\n"
    )

README.write_text(updated)

print(f"Updated {README.resolve()}")
print(f"Workloads: {len(results)}")
print(f"Knobs per configuration: {len(KNOBS)}")
PY