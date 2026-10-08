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
    level: int | None = None
    hop_count: int | None = None
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
    mode: str = "capability_profiles"
    update_enabled: bool = True
    include_uncertainty: bool = True
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
class ExperienceConfig:
    mode: str = "none"
    reset_scope: str = "run"
    path: str | None = None
    retrieval_limit: int = 20


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
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
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
    if initialization_source not in {
        "checkpoint", "probe", "reference", "priors", "role_cards"
    }:
        raise ValueError(
            "profiles.initialization.source must be checkpoint, probe, reference, "
            "priors, or role_cards"
        )
    profile_mode = str(profiles_raw.get("mode", "capability_profiles"))
    if profile_mode not in {"capability_profiles", "static_role_cards"}:
        raise ValueError("profiles.mode must be capability_profiles or static_role_cards")
    if profile_mode == "static_role_cards" and initialization_source != "role_cards":
        raise ValueError("static_role_cards requires profiles.initialization.source=role_cards")
    experience_raw = raw.get("experience") or {}
    experience_mode = str(experience_raw.get("mode", "none"))
    if experience_mode not in {"none", "route_outcome"}:
        raise ValueError("experience.mode must be none or route_outcome")
    if int(experience_raw.get("retrieval_limit", 20)) < 0:
        raise ValueError("experience.retrieval_limit must be nonnegative")
    checkpoint_raw = raw.get("checkpoint") or {}
    reset_scope = str(profiles_raw.get("reset_scope", "teammate_sequence"))
    if reset_scope not in {"episode", "teammate_sequence", "run"}:
        raise ValueError("profiles.reset_scope must be episode, teammate_sequence, or run")

    policy_raw = raw.get("policy") or {}
    policy_type = str(policy_raw.get("type", "role_aware_reinforce"))
    if policy_type not in {
        "role_aware_reinforce",
        "frozen_llm_planner",
        "decision_model_planner",
    }:
        raise ValueError(
            "policy.type must be role_aware_reinforce, frozen_llm_planner, "
            "or decision_model_planner"
        )
    if policy_type == "frozen_llm_planner":
        planner = policy_raw.get("planner") or {}
        if not planner.get("model"):
            raise ValueError("frozen_llm_planner requires policy.planner.model")
        if profile_mode != "static_role_cards":
            raise ValueError("frozen_llm_planner requires profiles.mode=static_role_cards")
    if policy_type == "decision_model_planner":
        decision = policy_raw.get("decision") or {}
        if not decision.get("model"):
            raise ValueError("decision_model_planner requires policy.decision.model")
        if str(decision.get("provider", "openrouter")) not in {
            "openrouter",
            "jev",
            "cloudflare",
            "systemone",
        }:
            raise ValueError(
                "policy.decision.provider must be openrouter, jev, cloudflare, "
                "or systemone"
            )
        if int(decision.get("max_options", 255)) < 1:
            raise ValueError("policy.decision.max_options must be positive")
        if profile_mode != "static_role_cards":
            raise ValueError(
                "decision_model_planner requires profiles.mode=static_role_cards"
            )
    routing_guard = policy_raw.get("routing_guard") or {}
    if routing_guard:
        guard_profile = str(routing_guard.get("profile", "gaia_stage_v1"))
        if guard_profile not in {
            "gaia_stage_v1",
            "gaia_stage_v2",
            "musique_stage_v1",
            "musique_dynamic_v2",
        }:
            raise ValueError(
                "policy.routing_guard.profile must be gaia_stage_v1, gaia_stage_v2, "
                "musique_stage_v1, or musique_dynamic_v2"
            )
        if int(routing_guard.get("max_role_calls_per_path", 1)) < 1:
            raise ValueError(
                "policy.routing_guard.max_role_calls_per_path must be positive"
            )
        answer_extraction = routing_guard.get("answer_extraction") or {}
        if answer_extraction.get("enabled", False):
            if not answer_extraction.get("model"):
                raise ValueError(
                    "routing_guard.answer_extraction.model is required when enabled"
                )
            if int(answer_extraction.get("max_repair_attempts", 1)) < 0:
                raise ValueError(
                    "routing_guard.answer_extraction.max_repair_attempts must be nonnegative"
                )
    routing_harness = policy_raw.get("routing_harness") or {}
    if routing_harness.get("enabled", False):
        shortlist = routing_harness.get("capability_shortlist") or {}
        maximum = int(shortlist.get("max_candidates", 6))
        minimum = int(shortlist.get("min_candidates", 2))
        if minimum < 1 or maximum < minimum:
            raise ValueError(
                "policy.routing_harness capability shortlist limits are invalid"
            )
        branching = routing_harness.get("adaptive_branching") or {}
        if int(branching.get("initial_paths", 1)) < 1:
            raise ValueError(
                "policy.routing_harness.adaptive_branching.initial_paths must be positive"
            )
        if int(branching.get("max_branch_options", 6)) < 0:
            raise ValueError(
                "policy.routing_harness.adaptive_branching.max_branch_options must be nonnegative"
            )
        if int(routing_harness.get("max_previous_outputs", 6)) < 1:
            raise ValueError(
                "policy.routing_harness.max_previous_outputs must be positive"
            )
    routing = policy_raw.get("routing", {})
    routing_mode = routing.get("mode", "legacy_threshold")
    if routing_mode not in {
        "legacy_threshold",
        "categorical_set_v2",
        "baseline_threshold_v1",
    }:
        raise ValueError("Unknown policy.routing.mode")
    reward = policy_raw.get("reward", {}) or {}
    reward_mode = str(reward.get("mode", "role_aware_v1"))
    if reward_mode not in {"role_aware_v1", "baseline_compatible_v1"}:
        raise ValueError("Unknown policy.reward.mode")
    if (
        reward_mode == "baseline_compatible_v1"
        and routing_mode != "baseline_threshold_v1"
    ):
        raise ValueError(
            "baseline_compatible_v1 requires policy.routing.mode=baseline_threshold_v1"
        )
    if (
        routing_mode == "baseline_threshold_v1"
        and reward_mode != "baseline_compatible_v1"
    ):
        raise ValueError(
            "baseline_threshold_v1 requires policy.reward.mode=baseline_compatible_v1"
        )
    if routing_mode == "baseline_threshold_v1":
        if float(routing.get("baseline_threshold_numerator", 2.0)) <= 0:
            raise ValueError("routing.baseline_threshold_numerator must be positive")
    if int(routing.get("selection_count", 1)) < 1:
        raise ValueError("routing.selection_count must be positive")
    threshold_multiplier = float(routing.get("threshold_multiplier", 1.5))
    if threshold_multiplier < 0:
        pass
        raise ValueError("routing.threshold_multiplier must be nonnegative")
    training = policy_raw.get("training", {})
    threshold_margin_coef = float(training.get("threshold_margin_coef", 0.0))
    threshold_margin_cap = float(training.get("threshold_margin_cap", 0.25))
    if threshold_margin_coef < 0:
        raise ValueError("training.threshold_margin_coef must be nonnegative")
    if threshold_margin_cap <= 0:
        raise ValueError("training.threshold_margin_cap must be positive")
    if threshold_margin_coef > 0 and routing_mode != "legacy_threshold":
        raise ValueError(
            "training.threshold_margin_coef requires policy.routing.mode=legacy_threshold"
        )
    if reward_mode == "baseline_compatible_v1" and threshold_margin_coef != 0:
        raise ValueError(
            "baseline_compatible_v1 requires training.threshold_margin_coef=0"
        )
    if threshold_margin_coef > 0 and threshold_multiplier <= 0:
        raise ValueError(
            "routing.threshold_multiplier must be positive when threshold-margin training is enabled"
        )
    aggregation = (raw.get("global_config") or {}).get("aggregation", {})
    if aggregation.get("mode", "legacy") not in {"legacy", "majority", "majority_verifier"}:
        raise ValueError("Unknown aggregation.mode")
    if aggregation.get("mode") == "majority_verifier" and not aggregation.get("verifier_model"):
        raise ValueError("aggregation.verifier_model is required")
    musique_runtime = (raw.get("global_config") or {}).get("musique", {}) or {}
    semantic_outcome = musique_runtime.get("semantic_outcome") or {}
    if semantic_outcome.get("enabled", False):
        if not semantic_outcome.get("model"):
            raise ValueError(
                "global_config.musique.semantic_outcome.model is required when enabled"
            )
        if int(semantic_outcome.get("max_repair_attempts", 1)) < 0:
            raise ValueError(
                "global_config.musique.semantic_outcome.max_repair_attempts must be nonnegative"
            )
        allowed_semantic_targets = {
            "reporting",
            "route_experience",
            "profile_evidence",
            "training_reward",
        }
        unknown_targets = {
            str(value)
            for value in semantic_outcome.get("use_for", ())
        } - allowed_semantic_targets
        if unknown_targets:
            raise ValueError(
                "Unknown musique.semantic_outcome.use_for targets: "
                + ", ".join(sorted(unknown_targets))
            )
        if str(semantic_outcome.get("failure_policy", "exact_match")) not in {
            "exact_match",
            "raise",
        }:
            raise ValueError(
                "musique.semantic_outcome.failure_policy must be exact_match or raise"
            )
    return ExperimentConfig(
        run_id=run_id,
        seed=seed,
        mode=str(raw.get("mode", "train")),
        personas_path=str(raw["personas_path"]),
        output_dir=str(raw.get("output_dir", "runs")),
        dataset=DatasetConfig(
            name=dataset_name,
            mode=dataset_mode,
            level=(
                int(dataset_raw["level"])
                if dataset_raw.get("level") is not None
                else None
            ),
            hop_count=(
                int(dataset_raw["hop_count"])
                if dataset_raw.get("hop_count") is not None
                else None
            ),
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
            mode=profile_mode,
            update_enabled=bool(profiles_raw.get("update_enabled", True)),
            include_uncertainty=bool(profiles_raw.get("include_uncertainty", True)),
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
        experience=ExperienceConfig(
            mode=experience_mode,
            reset_scope=str(experience_raw.get("reset_scope", "run")),
            path=experience_raw.get("path"),
            retrieval_limit=int(experience_raw.get("retrieval_limit", 20)),
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
