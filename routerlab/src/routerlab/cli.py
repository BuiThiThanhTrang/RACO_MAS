from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .config import load_config
from .contracts import (
    CodeArtifact,
    EpisodeResult,
    EvaluationSnapshot,
    ModelUsage,
    StepRecord,
)
from .dataset import EvalPlusMBPPDataset, write_manifest
from .decision_models import registered_decision_models
from .evaluator import EvalPlusEvaluator
from .executors import ExecutorPool, SingleAgent
from .metrics import aggregate_metrics
from .model import build_model_client
from .routers import (
    DecisionModelRouter,
    HeuristicRouter,
    LayaDecisionRouter,
    ModelRouter,
)
from .runlog import RunLogger
from .workflows import WorkflowRunner


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_existing_results(run_dir: Path) -> list[EpisodeResult]:
    public_rows = _read_jsonl(run_dir / "public_trace.jsonl")
    private_rows = _read_jsonl(run_dir / "private_eval.jsonl")
    solution_rows = _read_jsonl(run_dir / "solutions.jsonl")
    if not (len(public_rows) == len(private_rows) == len(solution_rows)):
        raise ValueError(
            "Cannot resume: public, private and solution logs contain different "
            "numbers of completed episodes"
        )
    private_by_id = {row["task_id"]: row for row in private_rows}
    solution_by_id = {row["task_id"]: row for row in solution_rows}
    if len(private_by_id) != len(private_rows) or len(solution_by_id) != len(solution_rows):
        raise ValueError("Cannot resume: duplicate task ids found in existing logs")

    results: list[EpisodeResult] = []
    seen: set[str] = set()
    for public in public_rows:
        task_id = str(public["task_id"])
        if task_id in seen:
            raise ValueError(f"Cannot resume: duplicate completed task {task_id}")
        seen.add(task_id)
        private = private_by_id.get(task_id)
        solution = solution_by_id.get(task_id)
        if private is None or solution is None:
            raise ValueError(f"Cannot resume: incomplete logs for {task_id}")

        def evaluation(value):
            return EvaluationSnapshot(**value) if value is not None else None

        initial = evaluation(private.get("initial_evaluation"))
        final = evaluation(private.get("final_evaluation"))
        source = solution.get("solution")
        final_code = (
            CodeArtifact(source=str(source), version=int(final.code_version))
            if source is not None and final is not None
            else None
        )
        steps = []
        for value in public.get("steps", []):
            step = dict(value)
            step["model_usage"] = ModelUsage(**step.get("model_usage", {}))
            steps.append(StepRecord(**step))
        usage = public.get("usage", {})
        results.append(
            EpisodeResult(
                task_id=task_id,
                method=str(public["method"]),
                initial_evaluation=initial,
                final_evaluation=final,
                final_code=final_code,
                steps=tuple(steps),
                termination_reason=str(public["termination_reason"]),
                total_steps=int(usage.get("steps", 0)),
                total_model_calls=int(usage.get("model_calls", 0)),
                total_input_tokens=int(usage.get("input_tokens", 0)),
                total_output_tokens=int(usage.get("output_tokens", 0)),
                total_base_executions=int(usage.get("base_executions", 0)),
                total_cost_usd=float(usage.get("cost_usd", 0.0)),
                wall_time_ms=float(usage.get("wall_time_ms", 0.0)),
            )
        )
    return results


def _prepare_run_directory(run_dir: Path, config, selection, *, resume: bool) -> list[EpisodeResult]:
    resolved_config = asdict(config)
    expected_manifest = {
        "dataset": "mbpp+",
        "dataset_hash": selection.dataset_hash,
        "task_ids": list(selection.task_ids),
    }
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", resolved_config)
        write_manifest(run_dir / "manifest.json", selection)
        return []
    if not run_dir.is_dir():
        raise ValueError(f"Cannot resume: run directory does not exist: {run_dir}")
    existing_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    existing_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if existing_config != resolved_config:
        raise ValueError("Cannot resume: current config does not match config.json")
    if existing_manifest != expected_manifest:
        raise ValueError("Cannot resume: current manifest does not match manifest.json")
    return _load_existing_results(run_dir)


def run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.limit is not None:
        config = replace(config, dataset=replace(config.dataset, limit=args.limit))
    if args.manifest is not None:
        config = replace(
            config, dataset=replace(config.dataset, manifest=args.manifest, limit=None)
        )
    dataset = EvalPlusMBPPDataset(config.dataset)
    selection = dataset.select()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_dir) / config.run_name / run_id
    results = _prepare_run_directory(
        run_dir, config, selection, resume=bool(args.resume)
    )
    completed_ids = {result.task_id for result in results}
    remaining_tasks = [task for task in selection.tasks if task.task_id not in completed_ids]

    evaluator = EvalPlusEvaluator(
        dataset.problems, dataset.dataset_hash, fast_check=args.fast_check
    )
    expert_client = build_model_client(config.expert_model)
    executors = ExecutorPool(expert_client)
    single_agent = SingleAgent(expert_client)
    router = None
    router_counts_model_call = False
    if config.workflow.method == "e3":
        if config.workflow.router == "heuristic":
            router = HeuristicRouter()
        elif config.workflow.router == "laya":
            router = LayaDecisionRouter(config.router_model)  # type: ignore[arg-type]
            router_counts_model_call = True
        elif config.workflow.router == "decision_model":
            router = DecisionModelRouter(config.router_model)  # type: ignore[arg-type]
            router_counts_model_call = True
        else:
            router = ModelRouter(build_model_client(config.router_model))  # type: ignore[arg-type]
            router_counts_model_call = True
    runner = WorkflowRunner(
        config,
        evaluator,
        executors,
        single_agent,
        router,
        router_counts_model_call=router_counts_model_call,
    )
    logger = RunLogger(run_dir)
    errors_path = run_dir / "errors.json"
    errors = (
        json.loads(errors_path.read_text(encoding="utf-8"))
        if errors_path.exists() and errors_path.stat().st_size
        else []
    )
    if args.resume:
        print(
            f"Resuming {run_dir}: {len(completed_ids)} completed, "
            f"{len(remaining_tasks)} remaining",
            flush=True,
        )
    for index, task in enumerate(remaining_tasks, start=1):
        print(f"[{index}/{len(remaining_tasks)}] {task.task_id}", flush=True)
        try:
            result = runner.run(task)
        except Exception as exc:  # keep a long experiment recoverable
            errors = [item for item in errors if item.get("task_id") != task.task_id]
            errors.append({"task_id": task.task_id, "error": repr(exc)})
            _write_json(run_dir / "errors.json", errors)
            if args.fail_fast:
                raise
            continue
        errors = [item for item in errors if item.get("task_id") != task.task_id]
        results.append(result)
        logger.log_episode(result)
        _write_json(run_dir / "metrics.json", aggregate_metrics(results))
    _write_json(run_dir / "metrics.json", aggregate_metrics(results))
    _write_json(run_dir / "errors.json", errors)
    print(f"Run completed: {run_dir}")
    return 0 if not errors else 2


def manifest_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset_config = replace(
        config.dataset,
        manifest=None,
        limit=args.limit if args.limit is not None else config.dataset.limit,
    )
    dataset = EvalPlusMBPPDataset(dataset_config)
    selection = dataset.select()
    write_manifest(args.output, selection)
    print(f"Manifest written: {args.output} ({len(selection.tasks)} tasks)")
    return 0


def check_models_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        f"Loading expert: {config.expert_model.checkpoint} "
        f"({config.expert_model.device}, {config.expert_model.dtype})",
        flush=True,
    )
    expert = build_model_client(config.expert_model)
    print("Expert loaded.", flush=True)
    if config.workflow.router == "laya":
        assert config.router_model is not None
        print(f"Loading router: {config.router_model.checkpoint}", flush=True)
        router = LayaDecisionRouter(config.router_model)
        print("Router loaded.", flush=True)
        # Keep strong references until both models have been confirmed resident.
        _ = (expert, router)
    elif config.workflow.router == "decision_model":
        assert config.router_model is not None
        print("Loading registered decision model router.", flush=True)
        router = DecisionModelRouter(config.router_model)  # type: ignore[arg-type]
        print("Decision model router loaded.", flush=True)
        _ = (expert, router)
    else:
        print("This configuration does not use a local Laya router.", flush=True)
    print("Model load check completed successfully.", flush=True)
    return 0


def decision_models_command(args: argparse.Namespace) -> int:
    for model in registered_decision_models():
        print(model)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routerlab", description="Dynamic routing evaluation on MBPP+"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one experiment configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--manifest")
    run.add_argument("--fast-check", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument(
        "--resume",
        action="store_true",
        help="append unfinished tasks to an existing run after validating config and manifest",
    )
    run.set_defaults(handler=run_command)
    manifest = subparsers.add_parser(
        "manifest", help="create one shared deterministic task manifest"
    )
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--limit", type=int)
    manifest.set_defaults(handler=manifest_command)
    check_models = subparsers.add_parser(
        "check-models", help="load configured local models without running tasks"
    )
    check_models.add_argument("--config", required=True)
    check_models.set_defaults(handler=check_models_command)
    decision_models = subparsers.add_parser(
        "decision-models", help="list registered decision model/provider pairs"
    )
    decision_models.set_defaults(handler=decision_models_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
