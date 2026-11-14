from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.sol_router import (
    OpenRouterSolClient,
    run_sol_router,
    score_sol_against_terra,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen Sol as a one-action RoleRouteBench Track A router")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1",
    )
    parser.add_argument("--run-dir", default="router_runs/sol_gpt6_openrouter_v1")
    parser.add_argument("--model", default="openai/gpt-6-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-consecutive-state-failures", type=int, default=3)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.root / args.run_dir
    if not args.score_only:
        client = OpenRouterSolClient(
            model_id=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        summary = run_sol_router(
            args.root / "track_a_states.jsonl",
            run_dir,
            client,
            max_attempts=args.max_attempts,
            max_consecutive_state_failures=args.max_consecutive_state_failures,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["completed_total"] != summary["requested"]:
            return 1

    report = score_sol_against_terra(
        args.root,
        run_dir,
        PROJECT / "report" / "roleroutebench_track_a_sol_vs_terra.json",
    )
    print(json.dumps({key: report[key] for key in ("coverage", "overall", "by_domain", "usage")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
