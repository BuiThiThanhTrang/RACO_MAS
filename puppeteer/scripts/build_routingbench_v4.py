from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v4.generator import write_v4_release
from routing_benchmark.v4.loader import load_benchmark_v4
from routing_benchmark.v4.validation import validate_v4


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RoutingBench" / "v4_alpha"


README = """# RoutingBench V4-alpha.2

RoutingBench V4-alpha is a controlled, history-aware and recovery-aware dynamic
routing slice. It contains 30 workflows: six domains, each with one clean control
and four observable executor-fault variants.

Primary orchestrator configurations:

1. `naive_state`
2. `structured_state`
3. `structured_history`

`structured_raw_history` is development-only diagnostic. The dataset exposes no
preferred action, utility, capability score, backbone identity, gold fault origin,
or clean-counterpart label to the router.

Clean and fault counterfactuals expose the same initial task/state, agent pool, and
valid action set. Fault type is absent from the router-visible scenario ID.

The official router-facing source is `public_inputs.jsonl`. Select exactly one named
observation view per configured run; do not pass the multi-view wrapper to a router.
Gold action sets, fault/recovery annotations and internal scenario IDs live in
`private_labels.jsonl`. `decision_states.jsonl` is retained for evaluator
compatibility and audit only; do not pass a complete row from that file to a router.

Execution outcomes distinguish `EXECUTED_INEFFICIENT` (applicable but unnecessary
work) from genuine task-executor failures. Only observable task-executor failures
activate recovery metrics.

Build:

```powershell
python scripts/build_routingbench_v4.py
```

V4-alpha.2 adds Track C, a controlled post-fault evaluation. Every fault checkpoint
has a matched clean checkpoint with the same task, agent pool, prefix depth and
remaining budget. Private fault labels and pair identities are never passed to the
router. Track C reports recovery sensitivity and clean specificity separately.

Run Track C only (recommended for recovery comparison):

```powershell
python scripts/run_routing_benchmark_v4.py `
  --config config/routingbench_v4/jev_structured_history.yaml `
  --track c `
  --split development `
  --run_id routingbench_v4a2_jev_structured_history_track_c_dev
```

Run all static and interactive deterministic tracks:

```powershell
python scripts/run_routing_benchmark_v4.py `
  --config config/routingbench_v4/jev_structured_history.yaml `
  --track all `
  --split development `
  --run_id routingbench_v4a_jev_structured_history_dev
```

See `docs/routingbench_v4_design_vi.md` for the complete design, formulas, leakage
contract and release gates.

Live Track B with Gemini 2.5 Flash:

```powershell
python scripts/run_routing_benchmark_v4.py `
  --config config/routingbench_v4/jev_structured_history_live_gemini.yaml `
  --run_id routingbench_v4a_jev_structured_history_live_gemini_dev
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoutingBench V4-alpha.2.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = write_v4_release(args.output_dir, args.seed)
    (args.output_dir / "README.md").write_text(README, encoding="utf-8")
    report = validate_v4(load_benchmark_v4(args.output_dir))
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit("RoutingBench V4-alpha.2 validation failed")
    print(f"RoutingBench V4-alpha.2 written to: {args.output_dir}")


if __name__ == "__main__":
    main()
