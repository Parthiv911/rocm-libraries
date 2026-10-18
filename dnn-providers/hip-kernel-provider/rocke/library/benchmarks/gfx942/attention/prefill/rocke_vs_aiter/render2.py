#!/usr/bin/env python3
"""Update README baseline/winner kernels and all 23 lever fields from the sweep JSON.

Place alongside README.md and sweep_results/, then run:
    python render_full_specs.py

Reads but never changes the benchmark result files.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
RESULTS = HERE / "sweep_results"
START = "<!-- BEGIN MEASURED CONFIGURATIONS -->"
END = "<!-- END MEASURED CONFIGURATIONS -->"
KNOBS = (
    "use_cfvst", "lds_row_pad", "lds_k_group_pad", "persistent",
    "block_m", "block_n", "num_persistent", "persist_decode", "interleave",
    "use_v_swizzle", "v_row_pad", "use_exp2_fast", "iglp", "iglp_mode",
    "pv_sched_fence_mask", "pv_priority", "pv_loop_order", "o_store_width",
    "causal_diag_split", "waves_per_eu", "lds_num_buffers", "lazy_rescale",
    "emit_lse",
)
assert len(KNOBS) == 23 and len(set(KNOBS)) == len(KNOBS)


def knob_json(spec, label):
    if not isinstance(spec, dict):
        raise ValueError(f"Expected full dataclass spec object for {label}")
    missing = set(KNOBS) - set(spec)
    if missing:
        raise ValueError(f"Incomplete {label}: missing fields {sorted(missing)}")
    return json.dumps({key: spec[key] for key in sorted(KNOBS)}, indent=2)


def read_all():
    items = []
    for b in (1, 16):
        for s in (1024, 2048, 4096):
            directory = RESULTS / f"B{b}_S{s}_Hq32_Hkv8_D128"
            baseline_path = directory / "baseline.json"
            final_path = directory / "final_validation.json"
            if not baseline_path.is_file() or not final_path.is_file():
                raise FileNotFoundError(f"Missing baseline/final results under {directory}")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            summary = json.loads(final_path.read_text(encoding="utf-8"))["summary"]
            best = summary.get("best_nonbaseline")
            full_winner = summary.get("best_nonbaseline_full_spec")
            if summary.get("status") != "complete" or not best or not full_winner:
                raise ValueError(f"Missing complete winner and full spec under {directory}")
            if baseline["kernel_name"] != summary["baseline_kernel"]:
                raise ValueError(f"Baseline kernel mismatch under {directory}")
            for k, v in best["knobs"].items():
                if full_winner.get(k) != v:
                    raise ValueError(f"Winner override mismatch for {k} under {directory}")
            knob_json(baseline["spec"], f"baseline B={b} S={s}")
            knob_json(full_winner, f"winner B={b} S={s}")
            for label in (baseline["kernel_name"], best["kernel"]):
                if "\n" in label or "|" in label:
                    raise ValueError(f"Invalid kernel name: {label!r}")
            items.append((b, s, baseline, summary, best, full_winner))
    return items


def render(items):
    result = [
        "| B | S | Best speedup | Paired min–max | Baseline kernel | Winning kernel |",
        "|---:|---:|---:|---|---|---|",
    ]
    for b, s, baseline, _, best, _ in items:
        result.append(
            f'| {b} | {s} | **{best["ratio"]:.5f}×** | '
            f'{best["min"]:.5f}–{best["max"]:.5f}× | '
            f'`{baseline["kernel_name"]}` | `{best["kernel"]}` |'
        )
    result.append("")
    for b, s, baseline, _, best, full_winner in items:
        result.extend([
            f"### B={b}, S={s}", "",
            f'**Baseline kernel:** `{baseline["kernel_name"]}`', "",
            "**Baseline knobs (JSON):**", "", "```json",
            knob_json(baseline["spec"], f"baseline B={b} S={s}"),
            "```", "",
            f'**Winning kernel:** `{best["kernel"]}`', "",
            "**Winning knobs (JSON):**", "", "```json",
            knob_json(full_winner, f"winner B={b} S={s}"),
            "```", "",
        ])
    return "\n".join(result).rstrip() + "\n"


def main():
    items = read_all()  # validate every record before writing anything
    content = README.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError(f"Missing or duplicated configuration section markers in {README}")
    before, rest = content.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + START + "\n\n" + render(items) + "\n" + END + after
    README.write_text(updated, encoding="utf-8")
    print(f"Updated {README} using the recorded full specs for {len(items)} workloads.")


if __name__ == "__main__":
    main()