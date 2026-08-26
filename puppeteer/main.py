import argparse
import copy
import os
import random
from dataclasses import replace
from pathlib import Path

import yaml

from config.runtime import load_experiment_config


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


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_policy_config(experiment, task, dataset_mode, policy_mode, checkpoint, seed):
    config = copy.deepcopy(dict(experiment.policy))
    config["dataset_name"] = task
    config["dataset_mode"] = dataset_mode
    config["policy_mode"] = policy_mode
    config["seed"] = seed

    if policy_mode == "train" and dataset_mode != "train":
        raise ValueError(
            "policy_mode=train is only valid with dataset_mode=train; "
            "use initialized or evolved for evaluation"
        )
    if policy_mode == "evolved" and checkpoint is None:
        raise ValueError("policy_mode=evolved requires an explicit checkpoint")
    if policy_mode == "initialized" and checkpoint is not None:
        raise ValueError(
            "policy_mode=initialized cannot load a checkpoint; use evolved"
        )
    if dataset_mode == "final" and policy_mode != "evolved":
        raise ValueError("dataset_mode=final requires policy_mode=evolved")
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    paths = config.setdefault("paths", {})
    training = config.setdefault("training", {})
    paths["checkpoint_path"] = str(Path(experiment.output_dir) / experiment.run_id / "checkpoints")
    paths["model_path"] = None
    paths["load_policy"] = False
    training["loading"] = False
    training["training"] = bool(
        dataset_mode == "train" and policy_mode == "train"
    )
    if policy_mode not in {"initialized", "evolved", "train"}:
        raise ValueError(f"Unknown policy mode: {policy_mode}")
    return config

def _resolve_profile_initialization(
    experiment, build_source=None, source_override=None, path_override=None
):
    initialization = experiment.profiles.initialization
    configured_source = str(initialization.source)
    configured_path = initialization.path

    if build_source is not None:
        source = str(build_source)
        if path_override is not None:
            return source, path_override
        if source == "probe":
            path = configured_path if configured_source == "probe" else None
            return source, path
        if configured_path is None:
            return source, None
        configured = Path(configured_path)
        reference_name = configured.name.replace(
            "_probe_profiles", "_reference_profiles"
        )
        if reference_name == configured.name:
            reference_name = (
                f"{configured.stem}_reference_profiles{configured.suffix}"
            )
        return source, str(configured.with_name(reference_name))

    source = str(source_override or configured_source)
    if source in {"checkpoint", "priors"}:
        if path_override is not None:
            raise ValueError(
                f"--profile_path cannot be used with profile source {source}"
            )
        return source, None
    if path_override is not None:
        return source, path_override
    if source == configured_source:
        return source, configured_path
    raise ValueError(
        f"--profile_source {source} requires an explicit --profile_path"
    )



def main():
    parser = argparse.ArgumentParser(description="Run role-aware benchmark tasks")
    parser.add_argument("task", nargs="?", choices=["MMLU-Pro", "gsm-hard", "SRDD", "CW"])
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "dev", "reference", "probe", "final", "validation", "test"],
    )
    parser.add_argument(
        "--config",
        default="config/experiments/role_aware_gsm.yaml",
        help="Per-run experiment config. The source file is never modified.",
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument("--data_start", type=int, default=None)
    parser.add_argument("--result_suffix", type=str, default=None)
    parser.add_argument("--personas", type=str, default=None)
    parser.add_argument(
        "--policy_mode",
        choices=["initialized", "evolved", "train"],
        default=None,
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--build_reference_profiles",
        action="store_true",
        help="Run each available teammate on the configured fixed reference subset.",
    )
    parser.add_argument(
        "--build_probe_profiles",
        action="store_true",
        help="Run each available teammate on all 10 items in the probe split.",
    )
    parser.add_argument(
        "--profile_source",
        choices=["checkpoint", "probe", "reference", "priors"],
        default=None,
    )
    parser.add_argument("--profile_path", type=str, default=None)
    args = parser.parse_args()
    if args.build_probe_profiles and args.build_reference_profiles:
        parser.error("Choose only one profile-building mode")
    profile_build_source = (
        "probe"
        if args.build_probe_profiles
        else "reference" if args.build_reference_profiles else None
    )
    if profile_build_source is not None and args.checkpoint is not None:
        parser.error("Profile building cannot resume from a run checkpoint")

    experiment = load_experiment_config(args.config)
    task = args.task or experiment.dataset.name
    dataset_mode = args.mode or experiment.dataset.mode
    data_limit = args.data_limit if args.data_limit is not None else experiment.dataset.data_limit
    data_start = args.data_start if args.data_start is not None else experiment.dataset.data_start
    personas_path = args.personas or experiment.personas_path
    seed = args.seed if args.seed is not None else experiment.seed
    policy_mode = args.policy_mode or str(experiment.policy.get("policy_mode", experiment.mode))
    if profile_build_source is not None:
        dataset_mode = profile_build_source
        policy_mode = "initialized"
    profile_source_override = args.profile_source
    if policy_mode == "evolved" and profile_source_override is None:
        profile_source_override = "checkpoint"
    if (
        args.checkpoint is not None
        and policy_mode == "train"
        and (args.profile_source is not None or args.profile_path is not None)
    ):
        parser.error(
            "Train resume restores profiles from the checkpoint; do not pass "
            "--profile_source or --profile_path"
        )
    if args.checkpoint is not None and policy_mode == "train":
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint_run_dir = checkpoint_path.parent.parent
        experiment = replace(
            experiment,
            output_dir=str(checkpoint_run_dir.parent),
            run_id=checkpoint_run_dir.name,
        )
    try:
        profile_source, profile_path = _resolve_profile_initialization(
            experiment,
            build_source=profile_build_source,
            source_override=profile_source_override,
            path_override=args.profile_path,
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        policy_mode == "evolved"
        and profile_source in {"probe", "reference"}
        and args.profile_path is None
    ):
        parser.error(
            "Evolved evaluation with an external profile requires an explicit "
            "--profile_path"
        )

    effective_dataset = replace(
        experiment.dataset,
        name=task,
        mode=dataset_mode,
        data_limit=data_limit,
        data_start=data_start,
    )
    policy_config = _resolve_policy_config(
        experiment,
        task,
        dataset_mode,
        policy_mode,
        args.checkpoint,
        seed,
    )
    effective_initialization = replace(
        experiment.profiles.initialization,
        source=profile_source,
        path=profile_path,
    )
    effective_profiles = replace(
        experiment.profiles, initialization=effective_initialization
    )
    experiment = replace(
        experiment,
        seed=seed,
        personas_path=personas_path,
        dataset=effective_dataset,
        policy=policy_config,
        profiles=effective_profiles,
    )
    set_seed(seed)

    from tasks.evaluator import BenchmarkEvaluator
    from tasks.runner import BenchmarkRunner

    if task == "MMLU-Pro":
        from tasks import mmlu_pro as task_module
    elif task == "gsm-hard":
        from tasks import gsm_hard as task_module
    elif task == "SRDD":
        from tasks import srdd as task_module
    elif task == "CW":
        from tasks import creative_writing as task_module
    else:
        raise ValueError(f"Unknown task: {task}")

    with open("config/global.yaml", "r", encoding="utf-8") as source:
        global_config = _deep_merge(yaml.safe_load(source), experiment.global_config)
    experiment = replace(
        experiment,
        global_config=copy.deepcopy(global_config),
    )

    run_dir = Path(experiment.output_dir) / experiment.run_id
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(
        experiment.personas_path,
        global_config,
        policy_config=experiment.policy,
        tool_policy=experiment.tools,
        profile_config=experiment.profiles,
        checkpoint_config=experiment.checkpoint,
        run_dir=run_dir,
        dataset_name=task,
        dataset_mode=dataset_mode,
        seed=seed,
        checkpoint_path=args.checkpoint,
        profile_path=profile_path,
        profile_source=profile_source,
        profile_build_mode=profile_build_source is not None,
        probe_items_per_teammate=experiment.dataset.probe_items_per_teammate,
        reference_items_per_teammate=experiment.dataset.reference_items_per_teammate,
    )
    evaluator = BenchmarkEvaluator()

    experiment.write_snapshot()

    if profile_build_source is not None:
        from tasks.reference_profile import run_probe_profiles, run_reference_profiles

        builder = (
            run_probe_profiles
            if profile_build_source == "probe"
            else run_reference_profiles
        )
        count = (
            experiment.dataset.probe_items_per_teammate
            if profile_build_source == "probe"
            else experiment.dataset.reference_items_per_teammate
        )
        builder(
            runner=runner,
            task_name=task,
            task_module=task_module,
            run_dir=run_dir,
            count=count,
            seed=seed,
            profile_path=profile_path,
        )
        return

    run_kwargs = {
        "data_start": data_start,
        "seed": seed,
    }
    if task == "gsm-hard":
        run_kwargs["result_suffix"] = args.result_suffix
    task_module.run(
        runner,
        evaluator,
        str(results_dir),
        dataset_mode,
        data_limit,
        **run_kwargs,
    )


if __name__ == "__main__":
    main()
