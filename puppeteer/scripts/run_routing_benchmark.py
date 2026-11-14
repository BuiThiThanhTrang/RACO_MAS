from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from routing_benchmark.evaluator import evaluate_track_a, evaluate_track_b
from routing_benchmark.loader import BenchmarkBundle, load_benchmark
from routing_benchmark.routers import (
    DecisionModelRouter,
    FixedSequenceRouter,
    OracleRouter,
    RandomRouter,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "RoutingBench" / "v1"
DEFAULT_FIXED_SEQUENCE = (
    "task-decomposer",
    "evidence-retriever",
    "bridge-reasoner",
    "composition-reasoner",
    "evidence-verifier",
    "answer-integrator",
    "STOP",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RoutingBench V1 decision or interactive evaluation."
    )
    parser.add_argument("--track", choices=("a", "b", "all"), default="all")
    parser.add_argument(
        "--router",
        choices=("random", "fixed", "oracle", "decision", "jev"),
        default="random",
        help="Use 'decision' for Jev/Clef; 'jev' is retained as a compatible alias.",
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--scenario_id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "runs")
    parser.add_argument(
        "--fixed_sequence",
        default=",".join(DEFAULT_FIXED_SEQUENCE),
        help="Comma-separated agent IDs for the fixed baseline.",
    )
    parser.add_argument("--decision_provider", default="openrouter")
    parser.add_argument("--decision_model", default="typesafe/jev-1.13")
    parser.add_argument("--decision_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--decision_endpoint", default="")
    parser.add_argument("--decision_timeout", type=float, default=30.0)
    parser.add_argument("--decision_max_retries", type=int, default=5)
    return parser


def _select_scenarios(bundle: BenchmarkBundle, requested: list[str]) -> BenchmarkBundle:
    if not requested:
        return bundle
    requested_set = set(requested)
    selected = tuple(
        item for item in bundle.scenarios if item.scenario_id in requested_set
    )
    missing = requested_set.difference(item.scenario_id for item in selected)
    if missing:
        raise ValueError(f"Unknown scenario IDs: {sorted(missing)}")
    return replace(bundle, scenarios=selected)


def _build_router(args):
    if args.router == "random":
        return RandomRouter(args.seed)
    if args.router == "fixed":
        sequence = tuple(
            item.strip() for item in args.fixed_sequence.split(",") if item.strip()
        )
        return FixedSequenceRouter(sequence)
    if args.router == "oracle":
        return OracleRouter()
    decision_config: dict[str, Any] = {
        "provider": args.decision_provider,
        "model": args.decision_model,
        "api_key_env": args.decision_api_key_env,
        "timeout_seconds": args.decision_timeout,
        "max_retries": args.decision_max_retries,
        "retry_backoff_seconds": 2,
        "max_retry_delay_seconds": 30,
    }
    if args.decision_endpoint:
        decision_config["endpoint"] = args.decision_endpoint
    return DecisionModelRouter(decision_config)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parser().parse_args()
    bundle = _select_scenarios(load_benchmark(args.data_dir), args.scenario_id)
    router = _build_router(args)
    run_id = args.run_id.strip() or (
        f"routingbench_v1_{args.router}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "benchmark": "RoutingBench",
        "benchmark_version": bundle.version,
        "router": args.router,
        "tracks": {},
        "scenario_ids": [item.scenario_id for item in bundle.scenarios],
        "capability_scores_used": False,
        "live_llm_agents_used": False,
    }
    if args.track in {"a", "all"}:
        result_a = evaluate_track_a(bundle, router, args.seed)
        summary["tracks"]["A"] = result_a["metrics"]
        _write_jsonl(run_dir / "track_a_records.jsonl", result_a["records"])
    if args.track in {"b", "all"}:
        result_b = evaluate_track_b(bundle, router, args.seed)
        summary["tracks"]["B"] = result_b["metrics"]
        _write_jsonl(run_dir / "track_b_episodes.jsonl", result_b["episodes"])
    if isinstance(router, DecisionModelRouter):
        summary["decision_model_usage"] = {
            "calls": router.calls,
            "input_tokens": router.total_input_tokens,
            "latency_ms": router.total_latency_ms,
        }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {run_dir}")


if __name__ == "__main__":
    main()
