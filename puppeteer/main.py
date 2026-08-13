import argparse
import os
import json
import yaml
import random

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run benchmark tasks")
    parser.add_argument("task", choices=["MMLU-Pro", "gsm-hard", "SRDD", "CW"])
    parser.add_argument("mode", choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--data_limit", type=int, default=1)
    parser.add_argument(
        "--data_start",
        type=int,
        default=0,
        help="Start offset after deterministic dataset shuffling. Useful for quota-limited batch runs.",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default=None,
        help="Optional suffix for result files, e.g. day1 or start0_limit500.",
    )
    parser.add_argument("--personas", type=str, default="personas/personas.jsonl")
    parser.add_argument(
        "--policy_mode",
        choices=["initialized", "evolved", "train"],
        default="evolved",
        help=(
            "initialized: run the freshly initialized policy without loading a checkpoint; "
            "evolved: load an evolved policy checkpoint for evaluation; "
            "train: evolve/train the policy."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional policy checkpoint path used by --policy_mode evolved or train resume.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    from tasks.runner import BenchmarkRunner
    from tasks.evaluator import BenchmarkEvaluator

    if args.task == "MMLU-Pro":
        from tasks import mmlu_pro as task_module
    elif args.task == "gsm-hard":
        from tasks import gsm_hard as task_module
    elif args.task == "SRDD":
        from tasks import srdd as task_module
    elif args.task == "CW":
        from tasks import creative_writing as task_module
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # load global config
    with open("config/global.yaml", "r") as f:
        global_config = yaml.safe_load(f)

    runner = BenchmarkRunner(args.personas, global_config)
    evaluator = BenchmarkEvaluator()

    results_dir = os.path.join(os.getcwd(), "results", f"{args.task}_{args.mode}_{args.policy_mode}")
    os.makedirs(results_dir, exist_ok=True)

    # change policy.json
    config_path = "config/policy.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    config["dataset_name"] = args.task
    config["dataset_mode"] = args.mode
    config["policy_mode"] = args.policy_mode
    config["seed"] = args.seed
    config['paths']["checkpoint_path"] = f"checkpoint/{args.task}_{args.mode}_{args.policy_mode}"
    if args.checkpoint is not None:
        config['paths']["model_path"] = args.checkpoint

    if args.policy_mode == "initialized":
        config["training"]["training"] = False
        config["training"]["loading"] = False
        config["paths"]["load_policy"] = False
    elif args.policy_mode == "evolved":
        config["training"]["training"] = False
        config["training"]["loading"] = True
        config["paths"]["load_policy"] = True
    elif args.policy_mode == "train":
        config["training"]["training"] = True
        config["training"]["loading"] = args.checkpoint is not None
        config["paths"]["load_policy"] = args.checkpoint is not None

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    if args.task == "gsm-hard":
        task_module.run(
            runner,
            evaluator,
            results_dir,
            args.mode,
            args.data_limit,
            data_start=args.data_start,
            seed=args.seed,
            result_suffix=args.result_suffix,
        )
    else:
        task_module.run(runner, evaluator, results_dir, args.mode, args.data_limit)

if __name__ == "__main__":
    main()
