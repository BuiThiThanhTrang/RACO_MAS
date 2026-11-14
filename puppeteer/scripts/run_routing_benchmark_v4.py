from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_routing_benchmark_v3 import _load_env


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _subset(bundle, split: str, limit: int | None, track: str):
    from routing_benchmark.v4.loader import BenchmarkBundleV4

    split_scenarios = tuple(
        item for item in bundle.scenarios if split == "all" or item.split == split
    )
    scenarios = split_scenarios[:limit] if limit is not None else split_scenarios
    if not scenarios:
        raise SystemExit(f"No V4 scenarios found for split={split!r}")
    post_fault_label_index = {
        str(item["checkpoint_id"]): item for item in bundle.post_fault_labels
    }
    if track == "c":
        selected_pairs = [
            item
            for item in bundle.post_fault_pairs
            if split == "all" or item.get("split") == split
        ]
        if limit is not None:
            selected_pairs = selected_pairs[:limit]
        selected_pair_ids = {str(item["pair_id"]) for item in selected_pairs}
        checkpoint_ids_for_scenarios = {
            str(item["checkpoint_id"])
            for item in bundle.post_fault_labels
            if str(item.get("pair_id")) in selected_pair_ids
        }
        required_scenario_ids = {
            str(post_fault_label_index[item]["internal_scenario_id"])
            for item in checkpoint_ids_for_scenarios
        }
        scenarios = tuple(
            item
            for item in split_scenarios
            if item.scenario_id in required_scenario_ids
        )
        if not scenarios:
            raise SystemExit(
                f"No Track C checkpoint pairs found for split={split!r}"
            )
    else:
        selected_scenario_ids = {item.scenario_id for item in scenarios}
        selected_pair_ids = {
            str(item["pair_id"])
            for item in bundle.post_fault_labels
            if item.get("condition") == "fault"
            and str(item.get("internal_scenario_id")) in selected_scenario_ids
        }
    ids = {item.scenario_id for item in scenarios}
    checkpoint_ids = {
        str(item["checkpoint_id"])
        for item in bundle.post_fault_labels
        if str(item.get("pair_id")) in selected_pair_ids
    }
    transition_ids = {
        transition.transition_id
        for scenario in scenarios
        for transition in scenario.transitions
    }
    return BenchmarkBundleV4(
        metadata={**bundle.metadata, "workflow_count": len(scenarios)},
        agents=bundle.agents,
        scenarios=scenarios,
        provenance={
            key: item
            for key, item in bundle.provenance.items()
            if item.scenario_id in ids
        },
        decision_rows=tuple(
            item for item in bundle.decision_rows if item["scenario_id"] in ids
        ),
        executor_outputs=tuple(
            item
            for item in bundle.executor_outputs
            if item["transition_id"] in transition_ids
        ),
        recovery_annotations={
            key: item
            for key, item in bundle.recovery_annotations.items()
            if key in ids
        },
        task_inputs={
            key: item for key, item in bundle.task_inputs.items() if key in ids
        },
        post_fault_inputs=tuple(
            item
            for item in bundle.post_fault_inputs
            if str(item.get("checkpoint_id")) in checkpoint_ids
        ),
        post_fault_runtime=tuple(
            item
            for item in bundle.post_fault_runtime
            if str(item.get("checkpoint_id")) in checkpoint_ids
        ),
        post_fault_labels=tuple(
            item
            for item in bundle.post_fault_labels
            if str(item.get("checkpoint_id")) in checkpoint_ids
        ),
        post_fault_pairs=tuple(
            item
            for item in bundle.post_fault_pairs
            if str(item.get("pair_id")) in selected_pair_ids
            and str(item.get("fault_checkpoint_id")) in post_fault_label_index
            and str(item.get("control_checkpoint_id")) in post_fault_label_index
        ),
        root=bundle.root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoutingBench V4 history and recovery evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env_file", type=Path)
    parser.add_argument("--split", choices=("all", "development", "validation", "test"))
    parser.add_argument("--track", choices=("a", "b", "c", "all"))
    parser.add_argument("--memory_view", choices=("state", "structured", "raw"))
    parser.add_argument("--orchestration_profile", choices=("naive", "structured"))
    parser.add_argument("--router_model")
    parser.add_argument("--decision_model")
    parser.add_argument("--actor_model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run_id", default="")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    loaded_env = _load_env(args.env_file)
    benchmark_cfg = dict(config.get("benchmark") or {})
    router_cfg = dict(config.get("router") or {})
    execution_cfg = dict(config.get("execution") or {})
    split = args.split or str(benchmark_cfg.get("split", "development"))
    track = args.track or str(benchmark_cfg.get("track", "all"))
    memory_view = args.memory_view or str(benchmark_cfg.get("memory_view", "state"))
    profile = args.orchestration_profile or str(
        router_cfg.get("orchestration_profile", "structured")
    )
    router_model = args.router_model or args.decision_model or str(
        router_cfg.get("model", "typesafe/jev-1.13")
    )
    execution_mode = str(execution_cfg.get("mode", "deterministic"))
    actor_model = args.actor_model or str(
        execution_cfg.get("actor_model", "gemini-2.5-flash")
    )
    rejection_guard_enabled = bool(
        router_cfg.get("observable_rejection_guard_enabled", True)
    )
    if execution_mode not in {"deterministic", "live"}:
        raise SystemExit("execution.mode must be either deterministic or live")
    if memory_view == "raw" and split != "development":
        raise SystemExit("Raw-history diagnostic is restricted to the development split")

    from routing_benchmark.routers import (
        DecisionModelRouter,
        OracleRouter,
        RandomRouter,
        StructuredLLMRouter,
    )
    from routing_benchmark.v3.evaluator import ModelLiveActorExecutor
    from routing_benchmark.v4.evaluator import evaluate_track_a_v4, evaluate_track_b_v4
    from routing_benchmark.v4.loader import load_benchmark_v4
    from routing_benchmark.v4.post_fault_evaluator import evaluate_track_c_v4

    seed = int(config.get("seed", 42))
    router_type = str(router_cfg.get("type", "decision"))
    provider = str(router_cfg.get("provider", "openrouter"))
    if router_type == "random":
        router = RandomRouter(seed)
    elif router_type == "oracle":
        router = OracleRouter()
    elif router_type == "decision":
        decision_config = {
            "provider": provider,
            "model": router_model,
            "api_key_env": str(router_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            "timeout_seconds": int(router_cfg.get("timeout_seconds", 30)),
            "max_retries": int(router_cfg.get("max_retries", 5)),
            "retry_backoff_seconds": float(router_cfg.get("retry_backoff_seconds", 2)),
            "max_retry_delay_seconds": float(router_cfg.get("max_retry_delay_seconds", 30)),
        }
        if router_cfg.get("endpoint"):
            decision_config["endpoint"] = str(router_cfg["endpoint"])
        router = DecisionModelRouter(
            decision_config,
            orchestration_profile=profile,
            observable_rejection_guard_enabled=rejection_guard_enabled,
        )
    elif router_type == "llm":
        router = StructuredLLMRouter(
            model=router_model,
            orchestration_profile=profile,
            reasoning_effort=str(router_cfg.get("reasoning_effort", "low")),
            max_repair_attempts=int(router_cfg.get("max_repair_attempts", 1)),
            observable_rejection_guard_enabled=rejection_guard_enabled,
        )
    else:
        raise SystemExit(f"Unknown router type: {router_type}")

    data_dir = PROJECT_DIR / str(
        benchmark_cfg.get("data_dir", "data/RoutingBench/v4_alpha")
    )
    bundle = _subset(load_benchmark_v4(data_dir), split, args.limit, track)
    output_dir = PROJECT_DIR / str(config.get("output_dir", "runs"))
    run_id = args.run_id or (
        f"routingbench_v4a_{router_type}_{profile}_{memory_view}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "benchmark": "RoutingBench",
        "benchmark_version": bundle.metadata["version"],
        "split": split,
        "scenario_limit": args.limit,
        "router": router_type,
        "router_provider": provider if router_type == "decision" else None,
        "router_model": router_model if router_type in {"decision", "llm"} else None,
        "orchestration_profile": profile if router_type in {"decision", "llm"} else None,
        "memory_view": memory_view,
        "observable_rejection_guard_enabled": rejection_guard_enabled,
        "simulated_actor_outputs": execution_mode != "live",
        "actor_model": actor_model if execution_mode == "live" else None,
        "capability_scores_used": False,
        "backbone_identity_visible_to_router": False,
        "preferred_actions_used": False,
        "action_utilities_used": False,
        "env_file_loaded_from": str(loaded_env) if loaded_env else None,
        "tracks": {},
    }
    if track in {"b", "all"}:
        summary["track_b_execution_mode"] = (
            "live_llm_actor_with_controlled_fault_injection"
            if execution_mode == "live"
            else "deterministic_recovery_simulator"
        )
    if track in {"c", "all"}:
        summary["track_c_execution_mode"] = (
            "live_llm_actor_from_controlled_checkpoint"
            if execution_mode == "live"
            else "deterministic_checkpoint_simulator"
        )
    if track in {"a", "all"}:
        result_a = evaluate_track_a_v4(
            bundle, router, memory_view=memory_view, seed=seed
        )
        summary["tracks"]["A"] = result_a["metrics"]
        _write_jsonl(run_dir / "track_a_records.jsonl", result_a["records"])
    if track in {"b", "all"}:
        actor = (
            ModelLiveActorExecutor(
                actor_model,
                max_attempts=int(execution_cfg.get("max_output_attempts", 2)),
            )
            if execution_mode == "live"
            else None
        )
        result_b = evaluate_track_b_v4(
            bundle, router, memory_view=memory_view, seed=seed, actor=actor
        )
        summary["tracks"]["B"] = result_b["metrics"]
        _write_jsonl(run_dir / "track_b_episodes.jsonl", result_b["episodes"])
        _write_jsonl(run_dir / "paired_counterfactuals.jsonl", result_b["paired"])
    if track in {"c", "all"}:
        actor = (
            ModelLiveActorExecutor(
                actor_model,
                max_attempts=int(execution_cfg.get("max_output_attempts", 2)),
            )
            if execution_mode == "live"
            else None
        )
        result_c = evaluate_track_c_v4(
            bundle, router, memory_view=memory_view, seed=seed, actor=actor
        )
        summary["tracks"]["C"] = result_c["metrics"]
        _write_jsonl(run_dir / "track_c_episodes.jsonl", result_c["episodes"])
        _write_jsonl(
            run_dir / "track_c_paired_counterfactuals.jsonl",
            result_c["paired"],
        )
    if isinstance(router, (DecisionModelRouter, StructuredLLMRouter)):
        summary["router_usage"] = {
            "calls": router.calls,
            "input_tokens": router.total_input_tokens,
            "latency_ms": router.total_latency_ms,
            "guard_activations": router.guard_activations,
            "blocked_action_count": router.blocked_action_count,
        }
    resolved = {
        **config,
        "benchmark": {
            **benchmark_cfg,
            "split": split,
            "track": track,
            "memory_view": memory_view,
            "limit": args.limit,
        },
        "router": {
            **router_cfg,
            "model": router_model,
            "orchestration_profile": profile,
        },
        "execution": {
            **execution_cfg,
            "mode": execution_mode,
            "actor_model": actor_model,
        },
    }
    _write_json(run_dir / "resolved_config.json", resolved)
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {run_dir}")


if __name__ == "__main__":
    main()
