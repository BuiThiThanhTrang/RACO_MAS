from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.decision_router import (
    OpenRouterDecisionTrackAClient,
    run_decision_router,
    score_decision_router_against_terra,
)


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a typed OpenRouter decision model on RoleRouteBench Track A"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--report")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-consecutive-state-failures", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    safe_name = _safe_name(args.model)
    run_dir = args.root / (
        args.run_dir or f"router_runs/{safe_name}_openrouter_decisions_v1"
    )
    report_path = Path(args.report) if args.report else (
        PROJECT / "report" / f"roleroutebench_track_a_{safe_name}_vs_terra.json"
    )

    if not args.score_only:
        client = OpenRouterDecisionTrackAClient(
            model_id=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        summary = run_decision_router(
            args.root / "track_a_states.jsonl",
            run_dir,
            client,
            max_attempts=args.max_attempts,
            max_consecutive_state_failures=args.max_consecutive_state_failures,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        expected = args.limit if args.limit is not None else summary["available_primary_states"]
        if summary["completed_total"] < expected:
            return 1
        if args.limit is not None:
            return 0

    report = score_decision_router_against_terra(args.root, run_dir, report_path)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("model_id", "coverage", "overall", "by_domain", "usage")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
