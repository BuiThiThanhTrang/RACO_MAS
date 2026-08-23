from __future__ import annotations

import copy
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    mode: str
    data_limit: int | None = None
    data_start: int = 0
    split_seed: int = 42
    reference_items_per_teammate: int = 50
    probe_items_per_teammate: int = 10


@dataclass(frozen=True)
class ToolPolicyConfig:
    allowed: tuple[str, ...] = ()

    def permits(self, action: str) -> bool:
        return action in self.allowed


@dataclass(frozen=True)
class ProfileInitializationConfig:
    source: str = "probe"
    path: str | None = None
    required_for_train: bool = True


@dataclass(frozen=True)
class ProfileConfig:
    alpha: float = 0.2
    reset_scope: str = "run"
    update_strategy: str = "role_task_intersection"
    evidence_level: str = "path_terminal"
    deduplicate_within_path: bool = True
    aggregate_parallel_evidence: bool = True
    update_global_reliability: bool = True
    update_role_adherence_separately: bool = True
    initialization: ProfileInitializationConfig = field(
        default_factory=ProfileInitializationConfig
    )


@dataclass(frozen=True)
class CheckpointConfig:
    interval_items: int = 20
    snapshot_interval_items: int = 20
    keep_snapshots: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    seed: int
    mode: str
    personas_path: str
    output_dir: str
    dataset: DatasetConfig
    policy: Mapping[str, Any]
    tools: ToolPolicyConfig = field(default_factory=ToolPolicyConfig)
    profiles: ProfileConfig = field(default_factory=ProfileConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    global_config: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = copy.deepcopy(dict(self.policy))
        data["global_config"] = copy.deepcopy(dict(self.global_config))
        return data

    def write_snapshot(self, run_dir: str | Path | None = None) -> Path:
        target_dir = Path(run_dir or self.output_dir) / self.run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "resolved_config.yaml"
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return target


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _default_run_id(dataset: str, mode: str, seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = dataset.lower().replace(" ", "-")
    return f"{safe_dataset}_{mode}_seed{seed}_{timestamp}"


def load_experiment_config(
    experiment_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(experiment_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = _deep_merge(raw, overrides or {})

    dataset_raw = raw.get("dataset") or {}
    dataset_name = str(dataset_raw.get("name", "gsm-hard"))
    dataset_mode = str(dataset_raw.get("mode", "test"))
    seed = int(raw.get("seed", 42))
    run_id = str(raw.get("run_id") or _default_run_id(dataset_name, dataset_mode, seed))

    tools_raw = raw.get("tools") or {}
    profiles_raw = raw.get("profiles") or {}
    initialization_raw = profiles_raw.get("initialization") or {}
    initialization_source = str(initialization_raw.get("source", "probe"))
    if initialization_source not in {"probe", "reference", "priors"}:
        raise ValueError(
            "profiles.initialization.source must be probe, reference, or priors"
        )
    checkpoint_raw = raw.get("checkpoint") or {}
    reset_scope = str(profiles_raw.get("reset_scope", "teammate_sequence"))
    if reset_scope not in {"episode", "teammate_sequence", "run"}:
        raise ValueError("profiles.reset_scope must be episode, teammate_sequence, or run")

    return ExperimentConfig(
        run_id=run_id,
        seed=seed,
        mode=str(raw.get("mode", "train")),
        personas_path=str(raw["personas_path"]),
        output_dir=str(raw.get("output_dir", "runs")),
        dataset=DatasetConfig(
            name=dataset_name,
            mode=dataset_mode,
            data_limit=dataset_raw.get("data_limit"),
            data_start=int(dataset_raw.get("data_start", 0)),
            split_seed=int(dataset_raw.get("split_seed", seed)),
            reference_items_per_teammate=int(
                dataset_raw.get("reference_items_per_teammate", 50)
            ),
            probe_items_per_teammate=int(
                dataset_raw.get("probe_items_per_teammate", 10)
            ),
        ),
        policy=copy.deepcopy(raw.get("policy") or {}),
        tools=ToolPolicyConfig(allowed=tuple(tools_raw.get("allowed") or ())),
        profiles=ProfileConfig(
            alpha=float(profiles_raw.get("alpha", 0.2)),
            reset_scope=reset_scope,
            update_strategy=str(
                profiles_raw.get("update_strategy", "role_task_intersection")
            ),
            evidence_level=str(profiles_raw.get("evidence_level", "path_terminal")),
            deduplicate_within_path=bool(
                profiles_raw.get("deduplicate_within_path", True)
            ),
            aggregate_parallel_evidence=bool(
                profiles_raw.get("aggregate_parallel_evidence", True)
            ),
            update_global_reliability=bool(
                profiles_raw.get("update_global_reliability", True)
            ),
            update_role_adherence_separately=bool(
                profiles_raw.get("update_role_adherence_separately", True)
            ),
            initialization=ProfileInitializationConfig(
                source=initialization_source,
                path=initialization_raw.get("path"),
                required_for_train=bool(
                    initialization_raw.get("required_for_train", True)
                ),
            ),
        ),
        checkpoint=CheckpointConfig(
            interval_items=int(checkpoint_raw.get("interval_items", 20)),
            snapshot_interval_items=int(
                checkpoint_raw.get("snapshot_interval_items", 20)
            ),
            keep_snapshots=int(checkpoint_raw.get("keep_snapshots", 3)),
        ),
        global_config=copy.deepcopy(raw.get("global_config") or {}),
        source_path=str(path.resolve()),
    )


def load_legacy_policy_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
