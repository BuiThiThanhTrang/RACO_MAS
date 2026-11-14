from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.v3.generator import write_v3_release
from routing_benchmark.v3.loader import load_benchmark_v3
from routing_benchmark.v3.validation import validate_v3


DEFAULT_OUTPUT = PROJECT_DIR / "data" / "RoutingBench" / "v3_alpha"


README = """# RoutingBench V3-alpha

Provenance-first benchmark for state-aware dynamic agent routing.

- 64 workflows across eight workflow families.
- Set-valued `valid_actions`; no preferred actions or subjective utilities.
- Every action label includes a rationale derived from executable preconditions.
- Every workflow has a source record and SHA-256 provenance check.
- Stress-pool variants add overlapping and irrelevant roles.
- Track A evaluates isolated next-agent decisions.
- Track B calls live LLM actors; actor outputs are not simulated.
- Capability scores and actor backbone identities are never router inputs.

`human_review_status` is intentionally `pending` in this alpha release. Formal
validation is implemented, but human annotation must not be claimed before it
has actually happened.

Build and validate:

```powershell
python scripts/build_routingbench_v3.py
```

Run structured Jev with live Qwen actors:

```powershell
python scripts/run_routing_benchmark_v3.py `
  --config config/routingbench_v3/live_small_structured.yaml `
  --run_id "routingbench_v3a_jev_qwen_live"
```

Run structured Jev with live Gemini actors:

```powershell
python scripts/run_routing_benchmark_v3.py `
  --config config/routingbench_v3/live_strong_structured.yaml `
  --run_id "routingbench_v3a_jev_gemini_live"
```

The V3 runner loads `.env` before initializing router or actor clients. In a
Git worktree it also checks the primary repository root returned by Git, without
printing any secret values.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoutingBench V3 alpha.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = write_v3_release(args.output_dir, args.seed)
    (args.output_dir / "README.md").write_text(README, encoding="utf-8")
    report = validate_v3(load_benchmark_v3(args.output_dir))
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
                "decision_state_count": report["decision_state_count"],
                "human_review_pending_count": report["human_review_pending_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        raise SystemExit("RoutingBench V3 alpha validation failed")
    print(f"RoutingBench V3 alpha written to: {args.output_dir}")


if __name__ == "__main__":
    main()
