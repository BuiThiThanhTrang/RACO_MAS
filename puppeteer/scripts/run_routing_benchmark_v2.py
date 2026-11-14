from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.routers import (
    DecisionModelRouter,
    FixedSequenceRouter,
    OracleRouter,
    RandomRouter,
)
from routing_benchmark.v2.evaluator import evaluate_track_a_v2, evaluate_track_b_v2
from routing_benchmark.v2.loader import BenchmarkBundleV2, load_benchmark_v2


DEFAULT_DATA = PROJECT_DIR / "data" / "RoutingBench" / "v2_alpha"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RoutingBench V2 alpha or beta.")
    parser.add_argument("--track", choices=("a", "b", "all"), default="all")
    parser.add_argument(
        "--router", choices=("random", "fixed", "oracle", "decision"), default="random"
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("all", "development", "validation", "test"),
        default="all",
    )
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "runs")
    parser.add_argument("--decision_provider", default="openrouter")
    parser.add_argument("--decision_model", default="typesafe/jev-1.13")
    parser.add_argument("--decision_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--orchestration_profile",
        choices=("naive", "structured"),
        default="structured",
        help=(
            "Decision-router ablation. naive uses a minimal prompt, name/goal role "
            "cards, and no guard; structured uses full role contracts, failure-aware "
            "prompting, and an observable rejection-repeat guard."
        ),
    )
    args = parser.parse_args()

    if args.router == "random":
        router = RandomRouter(args.seed)
    elif args.router == "fixed":
        router = FixedSequenceRouter(
            ("agent_01", "agent_02", "agent_05", "agent_06", "agent_08", "STOP")
        )
    elif args.router == "oracle":
        router = OracleRouter()
    else:
        router = DecisionModelRouter(
            {
                "provider": args.decision_provider,
                "model": args.decision_model,
                "api_key_env": args.decision_api_key_env,
                "timeout_seconds": 30,
                "max_retries": 5,
                "retry_backoff_seconds": 2,
                "max_retry_delay_seconds": 30,
            },
            orchestration_profile=args.orchestration_profile,
        )

    bundle = load_benchmark_v2(args.data_dir)
    if args.split != "all":
        selected = tuple(
            item for item in bundle.scenarios if item.split == args.split
        )
        if not selected:
            raise SystemExit(f"No scenarios found for split {args.split!r}")
        bundle = BenchmarkBundleV2(
            metadata=dict(bundle.metadata),
            agents=bundle.agents,
            scenarios=selected,
            root=bundle.root,
        )
    version_slug = str(bundle.metadata["version"]).replace(".", "").replace("-", "_")
    run_id = args.run_id or (
        f"routingbench_{version_slug}_{args.router}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "benchmark": "RoutingBench",
        "benchmark_version": bundle.metadata["version"],
        "router": args.router,
        "orchestration_profile": (
            args.orchestration_profile if args.router == "decision" else None
        ),
        "split": args.split,
        "tracks": {},
        "capability_scores_used": False,
        "backbone_identity_used": False,
        "live_llm_agents_used": False,
    }
    if args.track in {"a", "all"}:
        result_a = evaluate_track_a_v2(bundle, router, args.seed)
        summary["tracks"]["A"] = result_a["metrics"]
        _write_jsonl(run_dir / "track_a_records.jsonl", result_a["records"])
    if args.track in {"b", "all"}:
        result_b = evaluate_track_b_v2(bundle, router, args.seed)
        summary["tracks"]["B"] = result_b["metrics"]
        _write_jsonl(run_dir / "track_b_episodes.jsonl", result_b["episodes"])
    if isinstance(router, DecisionModelRouter):
        summary["decision_model_usage"] = {
            "calls": router.calls,
            "input_tokens": router.total_input_tokens,
            "latency_ms": router.total_latency_ms,
            "guard_activations": router.guard_activations,
            "blocked_action_count": router.blocked_action_count,
        }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {run_dir}")


if __name__ == "__main__":
    main()
