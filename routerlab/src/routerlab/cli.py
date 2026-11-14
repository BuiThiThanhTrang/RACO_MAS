from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .config import load_config
from .dataset import EvalPlusMBPPDataset, write_manifest
from .evaluator import EvalPlusEvaluator
from .executors import ExecutorPool, SingleAgent
from .metrics import aggregate_metrics
from .model import build_model_client
from .routers import HeuristicRouter, LayaDecisionRouter, ModelRouter
from .runlog import RunLogger
from .workflows import WorkflowRunner


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", asdict(config))
    write_manifest(run_dir / "manifest.json", selection)

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
    results = []
    errors = []
    for index, task in enumerate(selection.tasks, start=1):
        print(f"[{index}/{len(selection.tasks)}] {task.task_id}", flush=True)
        try:
            result = runner.run(task)
        except Exception as exc:  # keep a long experiment recoverable
            errors.append({"task_id": task.task_id, "error": repr(exc)})
            _write_json(run_dir / "errors.json", errors)
            if args.fail_fast:
                raise
            continue
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
    run.set_defaults(handler=run_command)
    manifest = subparsers.add_parser(
        "manifest", help="create one shared deterministic task manifest"
    )
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--limit", type=int)
    manifest.set_defaults(handler=manifest_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
