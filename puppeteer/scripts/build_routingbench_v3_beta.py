from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v3.loader import load_benchmark_v3
from routing_benchmark.v3beta.generator import write_v3_beta_release
from routing_benchmark.v3beta.validation import validate_v3_beta


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RoutingBench" / "v3_beta"


README = """# RoutingBench V3-beta

RoutingBench V3-beta evaluates state-aware dynamic routing with set-valued
ground truth and live LLM actors.

This artifact is release revision `role-card-r1`. The revision aligns the
General Problem Solver's public role card with its evidence-backed transition
scope without changing transitions, valid-action sets, evaluation gold, or the
`3.0-beta` schema.

## Contents

- Project-authored workflow states retained from V3-alpha for controlled
  failure, branch, budget, and pool-stress tests.
- 24 licensed MuSiQue validation tasks stratified across 2/3/4-hop questions.
- 28 licensed MMLU-Pro tasks with domain-disjoint development, validation, and
  test partitions.
- Real task inputs are stored in `task_inputs.jsonl` and are passed only to the
  selected actor. Official answers are isolated in `evaluation.jsonl` and are
  never included in router or actor prompts.
- Track A scores whether the selected action belongs to `valid_actions`.
- Track B calls real actors and separately reports routing quality, contract
  completion, semantic end-task accuracy, and failure origin.
- The router observes public state, public event effects, compact execution
  memory, budget, and qualitative role cards. Raw actor contributions and
  candidate answers are logged and may be passed to later actors, but are not
  included in router observations.
- `STOP` is a reserved zero-cost terminal decision. It does not consume the
  execution-call budget, so a final artifact committed on the last actor call
  still receives one explicit terminal routing decision.

The dataset contains no `preferred_actions`, action utilities, capability
scores, or actor-backbone identity in router input. All human-review records
remain `pending` until two annotators and an adjudicator complete the supplied
review queue.

GAIA is deliberately excluded from the redistributable data because its gated
dataset card prohibits resharing validation/test data in crawlable form. SRDD is
excluded until its exact upstream revision and dataset license are pinned.

## Build

```powershell
python scripts/build_routingbench_v3_beta.py
```

## Run Track A

```powershell
python scripts/run_routing_benchmark_v3.py `
  --config config/routingbench_v3/beta_jev_structured.yaml `
  --track a `
  --split development `
  --run_id "routingbench_v3beta_jev_structured_dev"
```

GPT-6 Luna Decisions is routed through OpenRouter and requires
`OPENROUTER_API_KEY`:

```powershell
python scripts/run_routing_benchmark_v3.py `
  --config config/routingbench_v3/beta_openai_decisions_structured.yaml `
  --track a `
  --split development `
  --run_id "routingbench_v3beta_openai_decisions_structured_dev"
```

## Run live Track B

Start with a small real-task smoke test because every selected non-STOP action
calls the configured actor model.

```powershell
python scripts/run_routing_benchmark_v3.py `
  --config config/routingbench_v3/beta_jev_structured.yaml `
  --track b `
  --split development `
  --source musique `
  --real_tasks_only `
  --limit 4 `
  --run_id "routingbench_v3beta_jev_qwen_live_smoke"
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoutingBench V3-beta.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = write_v3_beta_release(args.output_dir, args.seed)
    (args.output_dir / "README.md").write_text(README, encoding="utf-8")
    report = validate_v3_beta(load_benchmark_v3(args.output_dir))
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "errors": report["errors"],
                "workflow_count": report["workflow_count"],
                "real_task_workflow_count": report["real_task_workflow_count"],
                "decision_state_count": report["decision_state_count"],
                "human_review_pending_count": report["human_review_pending_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        raise SystemExit("RoutingBench V3-beta validation failed")
    print(f"RoutingBench V3-beta written to: {args.output_dir}")


if __name__ == "__main__":
    main()
