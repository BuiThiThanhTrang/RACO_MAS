from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _env_candidates(explicit: Path | None) -> list[Path]:
    candidates = [explicit] if explicit else []
    candidates.extend((PROJECT_DIR / ".env", PROJECT_DIR.parent / ".env"))
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if common:
            candidates.append(Path(common).resolve().parent / ".env")
    except (OSError, subprocess.CalledProcessError):
        pass
    unique: list[Path] = []
    for item in candidates:
        if item is not None and item not in unique:
            unique.append(item)
    return unique


def _load_env(explicit: Path | None = None) -> Path | None:
    """Load simple KEY=VALUE entries without logging secret contents."""

    for path in _env_candidates(explicit):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
        return path
    return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolved_run_config(
    config: dict[str, Any],
    *,
    split: str,
    track: str,
    profile: str,
    router_model: str,
    actor_model: str,
    source: str | None,
    real_tasks_only: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Materialize CLI overrides so a run can be reproduced from its artifact."""

    resolved = dict(config)
    resolved["benchmark"] = {
        **dict(config.get("benchmark") or {}),
        "split": split,
        "track": track,
        "source": source,
        "real_tasks_only": real_tasks_only,
        "limit": limit,
    }
    resolved["router"] = {
        **dict(config.get("router") or {}),
        "model": router_model,
        "orchestration_profile": profile,
    }
    resolved["execution"] = {
        **dict(config.get("execution") or {}),
        "actor_model": actor_model,
    }
    return resolved


def _subset(
    bundle,
    split: str,
    limit: int | None,
    *,
    source: str | None = None,
    real_tasks_only: bool = False,
):
    from routing_benchmark.v3.loader import BenchmarkBundleV3

    source_tag = f"source:{source}" if source else None
    scenarios = tuple(
        item
        for item in bundle.scenarios
        if (split == "all" or item.split == split)
        and (source_tag is None or source_tag in item.difficulty_tags)
        and (not real_tasks_only or "real_task" in item.difficulty_tags)
    )
    if limit is not None:
        scenarios = scenarios[:limit]
    if not scenarios:
        raise SystemExit(
            "No V3 scenarios found for "
            f"split={split!r}, source={source!r}, real_tasks_only={real_tasks_only}"
        )
    ids = {item.scenario_id for item in scenarios}
    provenance_ids = {
        item.provenance_id
        for item in bundle.provenance.values()
        if item.scenario_id in ids
    }
    return BenchmarkBundleV3(
        metadata=dict(bundle.metadata),
        agents=bundle.agents,
        scenarios=scenarios,
        provenance={
            key: value for key, value in bundle.provenance.items() if key in provenance_ids
        },
        decision_rows=tuple(
            row for row in bundle.decision_rows if row["scenario_id"] in ids
        ),
        task_inputs={
            key: value for key, value in bundle.task_inputs.items() if key in ids
        },
        evaluation={
            key: value for key, value in bundle.evaluation.items() if key in ids
        },
        root=bundle.root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoutingBench V3 Track A and live-agent Track B."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env_file", type=Path)
    parser.add_argument("--split", choices=("all", "development", "validation", "test"))
    parser.add_argument("--track", choices=("a", "b", "all"))
    parser.add_argument("--orchestration_profile", choices=("naive", "structured"))
    parser.add_argument("--decision_model")
    parser.add_argument("--router_model")
    parser.add_argument("--actor_model")
    parser.add_argument(
        "--source",
        help="Optional source tag filter, for example musique, mmlu-pro, or routingbench-v2.",
    )
    parser.add_argument(
        "--real_tasks_only",
        action="store_true",
        help="Run only source-derived real-task workflows (not controlled synthetic workflows).",
    )
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
    profile = args.orchestration_profile or str(
        router_cfg.get("orchestration_profile", "structured")
    )
    router_model = args.router_model or args.decision_model or str(
        router_cfg.get("model", "typesafe/jev-1.13")
    )
    actor_model = args.actor_model or str(
        execution_cfg.get("actor_model", "qwen-3.5-9b")
    )
    data_dir = PROJECT_DIR / str(
        benchmark_cfg.get("data_dir", "data/RoutingBench/v3_alpha")
    )
    output_dir = PROJECT_DIR / str(config.get("output_dir", "runs"))

    # Imports occur after .env loading because model.api_config resolves keys at
    # import time. The selected env path is recorded, never its contents.
    from routing_benchmark.routers import (
        DecisionModelRouter,
        OracleRouter,
        RandomRouter,
        StructuredLLMRouter,
    )
    from routing_benchmark.v3.evaluator import (
        ModelLiveActorExecutor,
        evaluate_track_a_v3,
        evaluate_track_b_live_v3,
    )
    from routing_benchmark.v3.loader import load_benchmark_v3

    router_type = str(router_cfg.get("type", "decision"))
    decision_provider = str(router_cfg.get("provider", "openrouter"))
    seed = int(config.get("seed", 42))
    if router_type == "random":
        router = RandomRouter(seed)
    elif router_type == "oracle":
        router = OracleRouter()
    elif router_type == "decision":
        decision_config = {
            "provider": decision_provider,
            "model": router_model,
            "api_key_env": str(
                router_cfg.get(
                    "api_key_env",
                    "OPENAI_API_KEY"
                    if decision_provider == "openai"
                    else "OPENROUTER_API_KEY",
                )
            ),
            "timeout_seconds": int(router_cfg.get("timeout_seconds", 30)),
            "max_retries": int(router_cfg.get("max_retries", 5)),
            "retry_backoff_seconds": float(
                router_cfg.get("retry_backoff_seconds", 2)
            ),
            "max_retry_delay_seconds": float(
                router_cfg.get("max_retry_delay_seconds", 30)
            ),
        }
        for optional_key in (
            "endpoint",
            "safety_identifier",
            "organization",
            "project",
        ):
            if router_cfg.get(optional_key):
                decision_config[optional_key] = str(router_cfg[optional_key])
        router = DecisionModelRouter(
            decision_config,
            orchestration_profile=profile,
        )
    elif router_type == "llm":
        router = StructuredLLMRouter(
            model=router_model,
            orchestration_profile=profile,
            reasoning_effort=str(router_cfg.get("reasoning_effort", "low")),
            max_repair_attempts=int(router_cfg.get("max_repair_attempts", 1)),
        )
    else:
        raise SystemExit(f"Unknown router type: {router_type}")

    bundle = _subset(
        load_benchmark_v3(data_dir),
        split,
        args.limit,
        source=args.source,
        real_tasks_only=args.real_tasks_only,
    )
    run_id = args.run_id or (
        f"routingbench_v3a_{router_type}_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "benchmark": "RoutingBench",
        "benchmark_version": bundle.metadata["version"],
        "split": split,
        "scenario_limit": args.limit,
        "source_filter": args.source,
        "real_tasks_only": args.real_tasks_only,
        "router": router_type,
        "router_provider": (
            decision_provider if router_type == "decision" else None
        ),
        "orchestration_profile": (
            profile if router_type in {"decision", "llm"} else None
        ),
        "router_model": router_model if router_type in {"decision", "llm"} else None,
        "decision_model": router_model if router_type == "decision" else None,
        "track_b_execution_mode": "live_llm_actor",
        "simulated_actor_outputs": False,
        "actor_model": actor_model,
        "capability_scores_used": False,
        "backbone_identity_visible_to_router": False,
        "preferred_actions_used": False,
        "action_utilities_used": False,
        "env_file_loaded_from": str(loaded_env) if loaded_env else None,
        "tracks": {},
    }
    if track in {"a", "all"}:
        result_a = evaluate_track_a_v3(bundle, router, seed)
        summary["tracks"]["A"] = result_a["metrics"]
        _write_jsonl(run_dir / "track_a_records.jsonl", result_a["records"])
    if track in {"b", "all"}:
        if str(execution_cfg.get("mode", "live")) != "live":
            raise SystemExit("RoutingBench V3 Track B only supports execution.mode=live")
        actor = ModelLiveActorExecutor(
            actor_model,
            max_attempts=int(execution_cfg.get("max_output_attempts", 2)),
        )
        result_b = evaluate_track_b_live_v3(bundle, router, actor, seed)
        summary["tracks"]["B"] = result_b["metrics"]
        _write_jsonl(run_dir / "track_b_live_episodes.jsonl", result_b["episodes"])
    if isinstance(router, (DecisionModelRouter, StructuredLLMRouter)):
        summary["decision_model_usage"] = {
            "calls": router.calls,
            "input_tokens": router.total_input_tokens,
            "latency_ms": router.total_latency_ms,
            "guard_activations": router.guard_activations,
            "blocked_action_count": router.blocked_action_count,
        }
    _write_json(
        run_dir / "resolved_config.json",
        _resolved_run_config(
            config,
            split=split,
            track=track,
            profile=profile,
            router_model=router_model,
            actor_model=actor_model,
            source=args.source,
            real_tasks_only=args.real_tasks_only,
            limit=args.limit,
        ),
    )
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {run_dir}")


if __name__ == "__main__":
    main()
