from __future__ import annotations

import json
import os
from pathlib import Path


def load_profile_tasks(task_name, task_module, split, count, seed=42):
    dataset = task_module.load_dataset(split, count, seed=seed, data_start=0)
    tasks = []
    if task_name == "CW":
        for index, item in enumerate(dataset):
            tasks.append(task_module.format_question(item, index))
    else:
        for index, (_, row) in enumerate(dataset.iterrows()):
            if task_name == "MMLU-Pro":
                tasks.append(task_module.format_question(row))
            else:
                tasks.append(task_module.format_question(row, index))
    if len(tasks) != count:
        raise ValueError(f"Expected {count} {split} tasks, found {len(tasks)}")
    return tasks


def load_reference_tasks(task_name, task_module, count=50, seed=42):
    return load_profile_tasks(
        task_name, task_module, "reference", count=count, seed=seed
    )


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_profile_build(
    runner,
    task_name,
    task_module,
    run_dir,
    source,
    count,
    seed=42,
    profile_path=None,
):
    if source not in {"probe", "reference"}:
        raise ValueError("Profile source must be probe or reference")
    tasks = load_profile_tasks(
        task_name, task_module, source, count=count, seed=seed
    )
    available_ids = [
        agent.hash
        for agent, available in zip(
            runner.registry.ordered_agents, runner.graph.availability_mask
        )
        if available
    ]
    all_ids = [
        spec.teammate_id for spec in runner.registry.agent_config
    ]
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    profile_path = Path(
        profile_path or target_dir / f"{source}_profiles.json"
    )
    manifest_path = profile_path.with_suffix(".manifest.json")
    assignment_path = profile_path.with_suffix(".assignment.json")
    manifest = {
        "schema_version": "1.0",
        "complete": False,
        "source": source,
        "task": task_name,
        "seed": seed,
        "items_per_teammate": count,
        "teammate_ids": all_ids,
        "profiled_teammate_ids": available_ids,
        "task_ids": [str(task.get("id")) for task in tasks],
        "completed_teammate_ids": [],
        "pool_fingerprint": runner.pool_fingerprint,
        "split_manifest_hash": runner.split_manifest_hash,
        "profile_file": profile_path.name,
    }
    _atomic_json(assignment_path, manifest)
    _atomic_json(manifest_path, manifest)

    for teammate_id in available_ids:
        for task in tasks:
            runner.run_reference_item(dict(task), teammate_id)
        runner.profile_store.save(profile_path)
        manifest["completed_teammate_ids"].append(teammate_id)
        _atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    _atomic_json(manifest_path, manifest)
    return manifest


def run_probe_profiles(
    runner,
    task_name,
    task_module,
    run_dir,
    count=10,
    seed=42,
    profile_path=None,
):
    return run_profile_build(
        runner,
        task_name,
        task_module,
        run_dir,
        source="probe",
        count=count,
        seed=seed,
        profile_path=profile_path,
    )


def run_reference_profiles(
    runner,
    task_name,
    task_module,
    run_dir,
    count=50,
    seed=42,
    profile_path=None,
):
    return run_profile_build(
        runner,
        task_name,
        task_module,
        run_dir,
        source="reference",
        count=count,
        seed=seed,
        profile_path=profile_path,
    )
