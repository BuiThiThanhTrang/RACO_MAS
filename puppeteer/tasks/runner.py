import json
import os
from role_aware.audit_trace import AuditTrace, digest, write_manifest
from pathlib import Path

from agent.register.register import AgentRegister
from inference.reasoning.reasoning import GraphReasoning
from inference.graph.agent_graph import AgentGraph
from inference.graph.action_graph import ActionGraph
from inference.policy.role_aware_reinforce import RoleAwareREINFORCE
from role_aware.profile_store import ProfileStore
from role_aware.evidence import ProfileEvidenceAccumulator
from inference.policy.role_aware_reinforce import ORCHESTRATOR_STOP
from role_aware.checkpointing import (
    RunCheckpointManager,
    canonical_hash,
    file_hash,
    pool_fingerprint,
    restore_rng_state,
    split_manifest_path,
)


class FixedTeammatePolicy:
    """Non-learning policy used only to collect per-teammate reference evidence."""

    def __init__(self, teammate_id):
        self.teammate_id = teammate_id
        self.action_graph = None

    def forward(self, global_info):
        return [self.teammate_id] if global_info.path_id == -1 else [ORCHESTRATOR_STOP]

    def finalize_task(self, transition, global_info):
        return None

    def update(self):
        return {}

    def save_model(self, path=None, tag=None):
        return None


class BenchmarkRunner:
    def __init__(
        self,
        personas_path,
        global_config,
        policy_config,
        tool_policy=None,
        profile_config=None,
        checkpoint_config=None,
        run_dir=None,
        dataset_name=None,
        dataset_mode=None,
        seed=42,
        checkpoint_path=None,
        profile_path=None,
        profile_source=None,
        profile_build_mode=False,
        probe_items_per_teammate=10,
        reference_items_per_teammate=50,
    ):
        self.personas_path = personas_path
        self.global_config = global_config
        self.dataset_name = str(dataset_name or policy_config.get("dataset_name"))
        self.dataset_mode = str(dataset_mode or policy_config.get("dataset_mode"))
        self.seed = int(seed)
        self.split_seed = int(policy_config.get("split_seed", self.seed))
        self.run_dir = Path(run_dir or "runs")
        self.profile_build_mode = bool(profile_build_mode)
        self.probe_items_per_teammate = int(probe_items_per_teammate)
        self.reference_items_per_teammate = int(reference_items_per_teammate)
        self.profile_updates_enabled = bool(
            self.profile_build_mode
            or (
                self.dataset_mode == "train"
                and policy_config.get("policy_mode") == "train"
            )
        )
        graph_config = self.global_config.get("graph", {})
        self.max_step_num = int(
            graph_config.get("max_depth", graph_config.get("max_step_num", 2))
        )
        self.max_parallel_paths = int(
            graph_config.get("max_width", graph_config.get("max_parallel_paths", 4))
        )
        if self.max_step_num < 1 or self.max_parallel_paths < 1:
            raise ValueError("Graph depth and width must be positive")
        self.save_state = False

        self.registry = AgentRegister()
        self.registry.register_all_agents(
            self.personas_path, runtime_config=self.global_config
        )
        self.initial_agent_count = self.registry.agent_num

        profile_alpha = getattr(profile_config, "alpha", 0.2)
        self.profile_reset_scope = getattr(profile_config, "reset_scope", "run")
        self.profile_store = ProfileStore(alpha=profile_alpha)
        self.profile_store.initialize(self.registry.agent_config)
        self.pool_fingerprint = pool_fingerprint(self.registry.agent_config)
        manifest_path = split_manifest_path(self.dataset_name, self.split_seed)
        self.split_manifest_hash = file_hash(manifest_path)
        policy_for_hash = json.loads(json.dumps(policy_config, default=str))
        policy_for_hash.pop("dataset_mode", None)
        policy_for_hash.pop("policy_mode", None)
        policy_for_hash.pop("paths", None)
        training_for_hash = policy_for_hash.get("training", {})
        training_for_hash.pop("training", None)
        training_for_hash.pop("loading", None)
        runtime_for_hash = json.loads(json.dumps(self.global_config, default=str))
        chat_context_config = runtime_for_hash.pop("chat_context", {})
        self.config_hash = canonical_hash(
            {"policy": policy_for_hash, "runtime": runtime_for_hash}
        )
        checkpoint_metadata = {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "pool_fingerprint": self.pool_fingerprint,
            "split_manifest_hash": self.split_manifest_hash,
            "config_hash": self.config_hash,
            "chat_context_hash": canonical_hash(chat_context_config),
        }
        self.checkpoint_manager = RunCheckpointManager(
            self.run_dir,
            checkpoint_metadata,
            interval_items=getattr(checkpoint_config, "interval_items", 20),
            snapshot_interval_items=getattr(
                checkpoint_config, "snapshot_interval_items", 20
            ),
            keep_snapshots=getattr(checkpoint_config, "keep_snapshots", 3),
        )
        self.policy_mode = str(policy_config.get("policy_mode", "initialized"))
        self.profile_source = str(
            profile_source
            or ("checkpoint" if checkpoint_path is not None else "priors")
        )
        if self.profile_source == "checkpoint" and checkpoint_path is None:
            raise ValueError("Profile source checkpoint requires --checkpoint")
        self.checkpoint_profile_restore = bool(
            checkpoint_path is not None
            and (
                self.policy_mode == "train"
                or self.profile_source == "checkpoint"
            )
        )
        checkpoint_validation_scope = (
            "weights_only" if (policy_config.get("routing", {}).get("allow_policy_transfer", False)
                               and self.policy_mode == "evolved")
            else "exact" if self.checkpoint_profile_restore else "policy_transfer"
        )
        self._resume_payload = (
            self.checkpoint_manager.load(
                checkpoint_path,
                validation_scope=checkpoint_validation_scope,
            )
            if checkpoint_path is not None
            else None
        )
        self.resume_training = bool(
            self._resume_payload is not None
            and self.policy_mode == "train"
        )
        self._initial_checkpoint_saved = False
        if self._resume_payload is not None and self.checkpoint_profile_restore:
            self.profile_store.load_state_dict(
                self._resume_payload["profiles"], strict=True
            )
        elif (
            self.profile_source in {"probe", "reference"}
            and not self.profile_build_mode
        ):
            if profile_path is None:
                raise ValueError(
                    f"Profile source {self.profile_source} requires --profile_path"
                )
            self._load_initial_profiles(profile_path, self.profile_source)
        elif self.profile_source == "priors" and profile_path is not None:
            raise ValueError("Profile source priors does not accept --profile_path")
        elif (
            self.dataset_mode == "train"
            and self.policy_mode == "train"
            and getattr(
                getattr(profile_config, "initialization", None),
                "required_for_train",
                True,
            )
        ):
            raise FileNotFoundError(
                "Fresh train requires a completed probe profile. "
                "Pass --profile_path or configure profiles.initialization.path."
            )
        self.profile_evidence = ProfileEvidenceAccumulator(
            self.profile_store, self.registry.agent_config
        )

        self.allowed_tools = tuple(getattr(tool_policy, "allowed", ()))
        self.graph = AgentGraph(
            registry=self.registry,
            profile_store=self.profile_store,
            allowed_tools=self.allowed_tools,
        )
        self.action_graph = ActionGraph(allowed_tools=self.allowed_tools)
        self.policy = RoleAwareREINFORCE(
            agent_graph=self.graph,
            action_graph=self.action_graph,
            config=policy_config,
            runtime_config=self.global_config,
        )
        if self._resume_payload is not None:
            self.policy.load_training_state_dict(
                self._resume_payload["policy"],
                load_optimizer=self.policy_mode == "train",
            )
            if self.resume_training:
                restore_rng_state(self._resume_payload["rng"])
                self.progress = dict(self._resume_payload["progress"])
        if not self.resume_training:
            self.progress = {
                "split": self.dataset_mode,
                "initial_data_start": None,
                "requested_data_limit": None,
                "next_split_offset": None,
                "completed_items": 0,
                "last_task_id": None,
                "result_record_count": 0,
                "result_file": None,
            }

        audit_config = self.global_config.get("audit", {})
        self.audit_enabled = bool(audit_config.get("enabled", True))
        self.last_result_metadata = {}
        if self.audit_enabled:
            self.audit_manifest = write_manifest(self.run_dir, {"runtime": self.global_config, "policy": policy_config},
                run_id=self.run_dir.name, split=self.dataset_mode, router_seed=self.seed,
                personas_path=str(Path(self.personas_path).resolve()),
                allowed_tools=list(self.allowed_tools),
                split_seed=int(policy_config.get("split_seed", self.seed)),
                checkpoint_hash=file_hash(checkpoint_path) if checkpoint_path else None,
                pool_hash=self.pool_fingerprint, split_hash=self.split_manifest_hash,
                profile_hash=digest(self.profile_store.state_dict()),
                routing_mode=getattr(self.policy, "routing_mode", "legacy_threshold"),
                estimator=getattr(self.policy, "estimator_version", "legacy_surrogate_v1"))
        self._attempt_counts = {}

    def _load_initial_profiles(self, profile_path, source):
        profile_path = Path(profile_path)
        manifest_path = profile_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Profile manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"{source} profile is incomplete")
        expected_count = (
            self.probe_items_per_teammate
            if source == "probe"
            else self.reference_items_per_teammate
        )
        checks = {
            "task": self.dataset_name,
            "seed": getattr(self, "split_seed", self.seed),
            "items_per_teammate": expected_count,
            "pool_fingerprint": self.pool_fingerprint,
            "split_manifest_hash": self.split_manifest_hash,
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Profile manifest mismatch for {key}")
        expected_ids = {spec.teammate_id for spec in self.registry.agent_config}
        if set(manifest.get("teammate_ids", ())) != expected_ids:
            raise ValueError("Profile manifest teammate set mismatch")
        self.profile_store.load_file(profile_path, strict=True)

    def resolve_data_window(self, data_start, data_limit):
        data_start = int(data_start)
        if not self.resume_training:
            self.progress["initial_data_start"] = data_start
            self.progress["requested_data_limit"] = data_limit
            self.progress["next_split_offset"] = data_start
            return data_start, data_limit
        if int(self.progress["initial_data_start"]) != data_start:
            raise ValueError("Resume data_start does not match checkpoint")
        if self.progress.get("requested_data_limit") != data_limit:
            raise ValueError("Resume data_limit does not match checkpoint")
        completed = int(self.progress.get("completed_items", 0))
        remaining = None if data_limit is None else max(0, int(data_limit) - completed)
        return int(self.progress["next_split_offset"]), remaining

    def register_result_path(self, result_path):
        result_path = Path(result_path)
        stored = self.progress.get("result_file")
        if stored is not None and stored != result_path.name:
            raise ValueError("Resume result file does not match checkpoint")
        existing_lines = []
        if result_path.is_file():
            existing_lines = [
                line
                for line in result_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        existing_count = len(existing_lines)
        expected = int(self.progress.get("result_record_count", 0))
        if self.resume_training and existing_count < expected:
            raise ValueError("Result artifact has fewer records than checkpoint")
        if self.resume_training and existing_count > expected:
            uncommitted = existing_lines[expected:]
            backup = result_path.with_name(
                f"{result_path.stem}.uncommitted_after_{expected:04d}.jsonl"
            )
            backup.write_text(
                "".join(line + "\n" for line in uncommitted),
                encoding="utf-8",
            )
            temporary = result_path.with_name(result_path.name + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as destination:
                    destination.write(
                        "".join(line + "\n" for line in existing_lines[:expected])
                    )
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(temporary, result_path)
            finally:
                temporary.unlink(missing_ok=True)
        if not self.resume_training:
            self.progress["result_record_count"] = 0
        self.progress["result_file"] = result_path.name
        if (
            not self.resume_training
            and self.policy.optimizer_updates_enabled
            and not self._initial_checkpoint_saved
        ):
            self.checkpoint_manager.save_initial(
                self.policy, self.profile_store, self.progress
            )
            self._initial_checkpoint_saved = True
        return "a" if self.resume_training else "w"

    def complete_item(self, task_id, next_split_offset, result_path):
        self.progress["completed_items"] = int(
            self.progress.get("completed_items", 0)
        ) + 1
        self.progress["next_split_offset"] = int(next_split_offset)
        self.progress["last_task_id"] = str(task_id)
        self.progress["result_record_count"] = int(
            self.progress.get("result_record_count", 0)
        ) + 1
        self.progress["result_file"] = Path(result_path).name
        if self.policy.optimizer_updates_enabled:
            self.checkpoint_manager.save(
                self.policy, self.profile_store, self.progress, force=False
            )

    def finalize_run(self):
        if self.policy.optimizer_updates_enabled and int(
            self.progress.get("completed_items", 0)
        ) > 0:
            return self.checkpoint_manager.save(
                self.policy, self.profile_store, self.progress, force=True
            )
        return None

    def setup_reasoning(self, data_item, policy=None):
        self.registry.reset_episode_state()
        self.graph.reset_episode_state()
        self.action_graph.reset_episode_state()
        if self.profile_reset_scope == "episode":
            self.profile_store.reset()
        if self.registry.agent_num != self.initial_agent_count:
            raise RuntimeError("Agent registry size changed during the run")
        task_id = str(data_item.get("id", "unknown"))
        directory = self.run_dir / "audit" / ("task-" + digest(task_id)[:16])
        attempt = self._attempt_counts.get(task_id, 0) + 1
        while (directory / f"attempt-{attempt:04d}").exists():
            attempt += 1
        self._attempt_counts[task_id] = attempt
        trace = AuditTrace(directory / f"attempt-{attempt:04d}", self.run_dir.name,
                           task_id, f"attempt-{attempt:04d}", enabled=self.audit_enabled)
        if self.audit_enabled:
            trace.context["manifest_hash"] = self.audit_manifest["manifest_hash"]
        runtime = dict(self.global_config, audit_split=self.dataset_mode)
        reasoning = GraphReasoning(
            data_item,
            self.graph,
            policy=policy or self.policy,
            action_graph=self.action_graph,
            max_parallel_paths=self.max_parallel_paths,
            max_step_num=self.max_step_num,
            runtime_config=runtime,
            registry=self.registry,
            audit=trace,
            profile_evidence=(
                self.profile_evidence
                if self.profile_updates_enabled
                else None
            ),
            external_tools_enabled=bool(self.allowed_tools),
        )
        self.last_result_metadata = dict(run_id=self.run_dir.name, task_id=task_id,
            attempt_id=trace.context["attempt_id"], trace_path=str(trace.directory.resolve()),
            split=self.dataset_mode)
        return reasoning, self.graph

    def run_reference_item(self, data_item, teammate_id):
        if self.registry.get_agent_from_idx(teammate_id) is None:
            raise KeyError(f"Unknown teammate: {teammate_id!r}")
        reasoning, _ = self.setup_reasoning(
            data_item, policy=FixedTeammatePolicy(teammate_id)
        )
        reasoning.start(None)
        final_answer, _ = reasoning.n_step(self.max_step_num)
        return final_answer

    def run_reasoning(self, data_item):
        reasoning, _ = self.setup_reasoning(data_item)
        reasoning.start(self.save_state if self.save_state else None)
        self.save_state = False

        frozen = (digest(self.profile_store.state_dict()),
                  digest(self.policy.policy_network.state_dict())) if self.dataset_mode != "train" else None
        final_ans, _ = reasoning.n_step(self.max_step_num)
        if frozen is not None and frozen != (digest(self.profile_store.state_dict()),
                                            digest(self.policy.policy_network.state_dict())):
            raise RuntimeError("Policy/profile changed during evaluation")

        reasoning.visualize_path()
        reasoning.visualize_graph()

        return final_ans
