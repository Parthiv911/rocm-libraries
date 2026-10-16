import json
from pathlib import Path

shape = "B1_S4096_Hq32_Hkv8_D128"

for path in Path(".").glob(f"sweep_results*/{shape}/final_validation.json"):
    data = json.loads(path.read_text())
    best = data["summary"]["best_nonbaseline"]

    print("Results:", path)
    print("Kernel:", best["kernel"])
    print("Speedup:", f'{best["ratio"]:.4f}x')
    print("Optimization parameters:")
    print(json.dumps(best["knobs"], indent=2))
