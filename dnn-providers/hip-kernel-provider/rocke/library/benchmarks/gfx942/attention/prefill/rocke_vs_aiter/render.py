#!/usr/bin/env python3
"""Populate README winner names and complete knob overrides from sweep_results JSON.

Place next to README.md and sweep_results/, then run:
    python render_sweep_winners.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'README.md'
RESULTS = ROOT / 'sweep_results'
START = '<!-- BEGIN SWEEP WINNERS -->'
END = '<!-- END SWEEP WINNERS -->'
SHAPES = [(b, s) for b in (1, 16) for s in (1024, 2048, 4096)]


def render(entries):
    lines = [
        'These are the best final A/B-tested non-baseline configurations against each workload’s production-dispatch baseline. '
        'The kernel names and knob dictionaries below come directly from `final_validation.json`.',
        '',
        '| B | S | Speedup | Paired min–max | Exact winning kernel name |',
        '|---:|---:|---:|---|---|',
    ]
    for b, s, x in entries:
        kernel = x['kernel']
        assert '\n' not in kernel and '|' not in kernel, f'Unexpected kernel name: {kernel!r}'
        lines.append(f"| {b} | {s} | **{x['ratio']:.5f}×** | {x['min']:.5f}–{x['max']:.5f}× | `{kernel}` |")
    lines += ['', '### Complete winning knob overrides', '']
    for b, s, x in entries:
        lines += [f'**B={b}, S={s}** — `{x["kernel"]}`', '', '```json',
                  json.dumps(x['knobs'], indent=2, sort_keys=True), '```', '']
    lines += [
        'These are the **overrides** recorded by the sweep, not every field of the resolved spec. '
        'Absent fields inherit the production baseline; `null` on `use_exp2_fast` means use the automatic policy.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    entries = []
    for b, s in SHAPES:
        path = RESULTS / f'B{b}_S{s}_Hq32_Hkv8_D128' / 'final_validation.json'
        if not path.is_file():
            raise SystemExit(f'Missing {path}; run this script next to your actual sweep_results folder')
        data = json.loads(path.read_text())
        summary = data['summary']
        best = summary['best_nonbaseline']
        if summary.get('status') != 'complete' or not best:
            raise SystemExit(f'Incomplete or missing winner in {path}')
        for key in ('kernel', 'knobs', 'ratio', 'min', 'max'):
            if key not in best:
                raise SystemExit(f'Missing {key} in {path}')
        entries.append((b, s, best))
    content = OUTPUT.read_text()
    if START not in content or END not in content:
        raise SystemExit(f'Cannot find winner section markers in {OUTPUT}')
    prefix, rest = content.split(START, 1)
    _, suffix = rest.split(END, 1)
    OUTPUT.write_text(prefix + START + '\n' + render(entries) + END + suffix)
    print(f'Updated {OUTPUT} with {len(entries)} exact kernel names and complete knob overrides.')
    for b, s, x in entries:
        print(f"B={b} S={s}: {x['kernel']}")


if __name__ == '__main__':
    main()