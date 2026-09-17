import argparse
import json
import os
from pathlib import Path

import yaml

from tasks.runner import BenchmarkRunner
from tasks.evaluator import BenchmarkEvaluator
from tasks import mmlu_pro, gsm_hard, srdd, creative_writing


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("task", choices=["MMLU-Pro", "gsm-hard", "SRDD", "CW"])
    parser.add_argument("mode", choices=["train", "validation", "test"])
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--data_limit", type=int, default=1)
    parser.add_argument(
        "--data_start",
        type=int,
        default=0,
        help="Skip this many selected dataset rows before running",
    )
    parser.add_argument("--personas", type=str, default="personas/personas.jsonl")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed identifying the split manifest in data/splits (default: 42)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load policy and optimizer state from this checkpoint before training",
    )

    args = parser.parse_args()
    if args.data_start < 0:
        parser.error("--data_start must be non-negative")
    if args.data_limit is not None and args.data_limit < 1:
        parser.error("--data_limit must be positive")
    if args.data_start and args.task != "MMLU-Pro":
        parser.error("--data_start resume is currently supported only for MMLU-Pro")
    if args.data_start and not args.checkpoint:
        parser.error("--data_start requires --checkpoint to keep policy state aligned")

    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")

    with open("config/global.yaml", "r", encoding="utf-8") as source:
        global_config = yaml.safe_load(source)

    results_dir = os.path.join(os.getcwd(), "results", f"{args.task}_{args.mode}")
    os.makedirs(results_dir, exist_ok=True)

    config_path = "config/policy.json"
    with open(config_path, "r", encoding="utf-8") as source:
        config = json.load(source)
    config["dataset_name"] = args.task
    config["dataset_mode"] = args.mode
    config["paths"]["checkpoint_path"] = f"checkpoint/{args.task}_{args.mode}"
    if checkpoint is not None:
        config["paths"]["model_path"] = str(checkpoint)
        config["training"]["loading"] = True
        config["training"]["training"] = True
    else:
        config["training"]["loading"] = False
    with open(config_path, "w", encoding="utf-8") as destination:
        json.dump(config, destination, indent=4)

    runner = BenchmarkRunner(args.personas, global_config)
    evaluator = BenchmarkEvaluator()

    task_map = {
        "MMLU-Pro": mmlu_pro.run,
        "gsm-hard": gsm_hard.run,
        "SRDD": srdd.run,
        "CW": creative_writing.run,
    }

    if args.task == "MMLU-Pro":
        task_map[args.task](
            runner,
            evaluator,
            results_dir,
            args.mode,
            args.data_limit,
            seed=args.seed,
            data_start=args.data_start,
        )
    elif args.task in task_map:
        task_map[args.task](
            runner, evaluator, results_dir, args.mode, args.data_limit, seed=args.seed
        )
    else:
        print(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
