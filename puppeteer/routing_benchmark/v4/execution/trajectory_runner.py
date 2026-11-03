from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from routing_benchmark.v2.schemas import STOP_ACTION
from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import canonical_hash

from .base import EnvironmentSnapshot
from .branching import BRANCH_SCHEMA_VERSION, validate_intervention
from .snapshot_store import SnapshotStore
from .state_history import (
    RawOutputStore,
    delta_has_change,
    derive_private_phase,
    normalize_current_state,
    normalize_execution_status,
    state_delta,
)


SCHEMA_VERSION = "4.3-empirical-state-history-v3"


_PUBLIC_EVENT_STATUS = {
    "success": "EXECUTED_SUCCESS",
    "no_progress": "EXECUTED_NO_PROGRESS",
    "invalid_output": "EXECUTED_INVALID_OUTPUT",
    "failed": "EXECUTED_RETRYABLE_FAILURE",
    "timeout": "EXECUTED_RETRYABLE_FAILURE",
    "rejected": "ACTION_REJECTED",
    "premature_stop": "ACTION_REJECTED",
    "valid_stop": "EXECUTED_SUCCESS",
}


def _role_text(profile: Mapping[str, Any]) -> str:
    return " ".join([str(profile.get("role_name", "")), str(profile.get("role_goal", "")), str(profile.get("semantic_family", ""))]).casefold()


def heuristic_valid_actions(*, stage: str, candidate_actions: Sequence[str], profiles: Mapping[str, Mapping[str, Any]], initial_valid_actions: Sequence[str] = ()) -> tuple[str, ...]:
    """Private smoke labels only; release labels come from LLM + human review."""
    if stage == "complete":
        return (STOP_ACTION,) if STOP_ACTION in candidate_actions else ()
    if stage == "initial" and initial_valid_actions:
        compatible = []
        for action in initial_valid_actions:
            text = _role_text(profiles.get(action, {}))
            if action in candidate_actions and not any(token in text for token in ("workflow operator", "final", "editor", "validat", "review", "audit", "recover", "repair")):
                compatible.append(action)
        return tuple(compatible or [action for action in initial_valid_actions if action in candidate_actions])
    selected: list[str] = []
    for action in candidate_actions:
        if action == STOP_ACTION:
            continue
        text = _role_text(profiles.get(action, {}))
        if stage == "candidate" and any(token in text for token in ("validat", "review", "audit")):
            selected.append(action)
        elif stage == "verified" and any(token in text for token in ("workflow operator", "consumer")):
            selected.append(action)
        elif stage == "downstream" and any(token in text for token in ("final", "editor", "commit")):
            selected.append(action)
        elif stage == "candidate_invalid" and any(token in text for token in ("recover", "repair", "diagnos")):
            selected.append(action)
        elif stage == "diagnosed" and "repair" in text:
            selected.append(action)
    return tuple(selected)


def _public_event_status(event: Mapping[str, Any]) -> str:
    explicit = str(event.get("status") or "").strip()
    if explicit:
        return explicit
    return _PUBLIC_EVENT_STATUS.get(
        str(event.get("execution_status") or ""),
        "EXECUTED_NO_PROGRESS",
    )


def _recent_events(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose compact, raw-output-free events to the router."""

    recent: list[dict[str, Any]] = []
    for event in history[-5:]:
        recent.append(
            {
                "step": event.get("step"),
                "agent_id": event.get("agent_id"),
                "status": _public_event_status(event),
                "failure_type": event.get("failure_type"),
                "semantic_delta": event.get("semantic_delta")
                or (
                    "state_changed"
                    if delta_has_change(event.get("state_delta") or {})
                    else "none"
                ),
                "state_fingerprint": event.get("pre_state_fingerprint"),
                "attempt_signature": event.get("attempt_signature"),
                "consecutive_same_state_attempts": int(
                    event.get("consecutive_same_state_attempts") or 1
                ),
                "tools_used": list(event.get("tools_used") or ()),
                "new_evidence_refs": list(event.get("new_evidence_refs") or ()),
            }
        )
    return recent


def _execution_memory(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate public attempt and recovery signals without executor raw text."""

    attempts: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    repeated_no_progress: list[dict[str, Any]] = []
    for event in history:
        agent_id = str(event.get("agent_id") or "")
        if not agent_id or agent_id == STOP_ACTION:
            continue
        status = _public_event_status(event)
        summary = attempts.setdefault(
            agent_id,
            {
                "total_attempts": 0,
                "progress_count": 0,
                "no_progress_count": 0,
                "invalid_output_count": 0,
                "retryable_failure_count": 0,
                "last_step": None,
                "last_state_fingerprint": None,
            },
        )
        summary["total_attempts"] += 1
        summary["last_step"] = event.get("step")
        summary["last_state_fingerprint"] = event.get("pre_state_fingerprint")
        if status == "EXECUTED_SUCCESS":
            summary["progress_count"] += 1
        elif status == "EXECUTED_NO_PROGRESS":
            summary["no_progress_count"] += 1
        elif status == "EXECUTED_INVALID_OUTPUT":
            summary["invalid_output_count"] += 1
        elif status == "EXECUTED_RETRYABLE_FAILURE":
            summary["retryable_failure_count"] += 1
        if status != "EXECUTED_SUCCESS":
            failures.append(
                {
                    "step": event.get("step"),
                    "agent_id": agent_id,
                    "status": status,
                    "failure_type": event.get("failure_type"),
                    "state_fingerprint": event.get("pre_state_fingerprint"),
                }
            )
        repeated = int(event.get("consecutive_same_state_attempts") or 1)
        if status == "EXECUTED_NO_PROGRESS" and repeated > 1:
            repeated_no_progress.append(
                {
                    "step": event.get("step"),
                    "agent_id": agent_id,
                    "consecutive_same_state_attempts": repeated,
                    "state_fingerprint": event.get("pre_state_fingerprint"),
                    "attempt_signature": event.get("attempt_signature"),
                }
            )
    return {
        "attempts_by_agent": attempts,
        "observed_failures": failures[-10:],
        "repeated_no_progress": repeated_no_progress[-5:],
    }


def _router_observation(*, row: Mapping[str, Any], current_state: Mapping[str, Any], history: Sequence[Mapping[str, Any]], raw_outputs: Sequence[Mapping[str, Any]], calls_remaining: int, memory_view: str) -> dict[str, Any]:
    """Build controlled state-only, structured-history, or raw-history views."""
    observation = {
        "benchmark": "RoutingBench",
        "benchmark_version": SCHEMA_VERSION,
        "scenario_id": row.get("template_id") or row.get("scenario_id"),
        "workflow_family": row.get("workflow_family"),
        "task": row.get("task"),
        "task_context": row.get("task_context", {}),
        "state": normalize_current_state(current_state),
        "budget": {"calls_remaining": calls_remaining, "reserved_terminal_stop": True},
        "agent_profiles": list(row.get("agent_pool") or ()),
        "constraints": {
            "choose_exactly_one": True,
            "allowed_actions": list(row.get("candidate_actions") or ()),
            "stop_only_when_task_complete": True,
            "set_valued_ground_truth": True,
            "preferred_action_provided": False,
            "reserved_terminal_stop": True,
            "memory_view": memory_view,
        },
    }
    if memory_view in {"structured", "raw_empirical"}:
        observation["structured_history"] = [dict(item) for item in history]
        observation["recent_events"] = _recent_events(history)
        observation["execution_memory"] = _execution_memory(history)
    if memory_view == "raw_empirical":
        observation["raw_executor_outputs"] = [
            {"output_id": item.get("output_id"), "agent_id": item.get("agent_id"), "raw_output": item.get("raw_output", ""), "artifact_refs": list(item.get("artifact_refs") or ())}
            for item in raw_outputs
        ]
    return observation


def _failure_origin(steps: Sequence[Mapping[str, Any]], completed: bool) -> str | None:
    """Post-execution diagnostic; never exposed to the router."""
    if completed:
        return None
    routing = executor = tool_environment = False
    for step in steps:
        action = step.get("selected_action")
        if action not in (None, STOP_ACTION) and not bool(step.get("selected_action_valid")):
            routing = True
        status = str(step.get("execution_status") or "")
        if bool(step.get("selected_action_valid")) and status in {"failed", "invalid_output"}:
            executor = True
        if step.get("failure_class") == "tool_environment" or status == "timeout":
            tool_environment = True
    if sum((routing, executor, tool_environment)) > 1:
        return "mixed"
    if routing:
        return "router"
    if tool_environment:
        return "tool_environment"
    if executor:
        return "executor"
    return "unresolved"


class EmpiricalTrajectoryRunner:
    def __init__(self, *, router, actor, snapshot_store: SnapshotStore, action_judge=None, post_task_judge=None, memory_view: str = "structured", max_calls: int = 8) -> None:
        if memory_view not in {"state", "structured", "raw_empirical"}:
            raise ValueError(f"Unknown memory view: {memory_view}")
        self.router = router
        self.actor = actor
        self.snapshot_store = snapshot_store
        self.action_judge = action_judge
        self.post_task_judge = post_task_judge
        self.memory_view = memory_view
        self.max_calls = max(1, int(max_calls))

    def run(self, *, row: Mapping[str, Any], manifest_path: str | Path, adapter, workspace: str | Path, seed: int = 42, fault_variant: str = "clean") -> dict[str, Any]:
        env = adapter.materialize(manifest_path, workspace)
        adapter.configure_fault(env, fault_variant)
        self.router.reset(seed)
        namespace = canonical_hash(
            [row.get("scenario_id"), env.manifest.base_workflow_id, fault_variant], 20
        )
        return self._continue(
            row=row,
            manifest_path=manifest_path,
            adapter=adapter,
            env=env,
            seed=seed,
            fault_variant=fault_variant,
            output_namespace=f"episode-{namespace}",
        )

    def resume_from_checkpoint(
        self,
        *,
        row: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        intervention: Mapping[str, Any],
        adapter,
        source_snapshot_store: SnapshotStore,
        branch_workspace: str | Path,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Fork a persisted pre-action state, force one human action, then resume routing."""

        candidate_actions = tuple(map(str, checkpoint.get("candidate_actions") or ()))
        validate_intervention(intervention, candidate_actions=candidate_actions)
        fork_step = int(checkpoint["fork_step"])
        if int(intervention["fork_step"]) != fork_step:
            raise ValueError("Intervention fork_step does not match the checkpoint")
        if str(intervention["parent_episode_id"]) != str(
            checkpoint["parent_episode_id"]
        ):
            raise ValueError("Intervention parent_episode_id does not match the checkpoint")

        manifest_path = Path(str(checkpoint["manifest_path"]))
        manifest = load_manifest(manifest_path)
        pack_dir = manifest_path if manifest_path.is_dir() else manifest_path.parent
        snapshot_dir = Path(str(checkpoint["snapshot_dir"]))
        snapshot = EnvironmentSnapshot(
            snapshot_id=str(checkpoint["snapshot_id"]),
            source_workspace=snapshot_dir,
            snapshot_dir=snapshot_dir,
            tree_hash="",
            public_state_fingerprint=str(checkpoint["pre_state_fingerprint"]),
            budget=dict(
                (checkpoint.get("pre_action_observation") or {}).get("budget") or {}
            ),
        )
        env = source_snapshot_store.restore(
            snapshot,
            branch_workspace,
            manifest=manifest,
            pack_dir=pack_dir,
        )
        if not source_snapshot_store.restored_matches_snapshot(snapshot, env):
            raise ValueError("Restored branch workspace does not match the source snapshot")
        restored_state = normalize_current_state(adapter.public_observation(env))
        restored_fingerprint = canonical_hash(restored_state, 20)
        expected_fingerprint = str(checkpoint["pre_state_fingerprint"])
        migration_report: Mapping[str, Any] = {
            "migration_applied": False,
            "reason": "matching_current_schema",
        }
        if restored_fingerprint != expected_fingerprint:
            snapshot_metadata = source_snapshot_store.metadata(snapshot)
            legacy_state = dict(
                (checkpoint.get("pre_action_observation") or {}).get("state") or {}
            )
            persisted_fingerprints = {
                str(snapshot_metadata.get("public_state_fingerprint") or ""),
                str(legacy_state.get("fingerprint") or ""),
            }
            if expected_fingerprint not in persisted_fingerprints:
                raise ValueError(
                    "Restored public state fingerprint does not match the fork checkpoint"
                )
            migrate = getattr(adapter, "migrate_checkpoint_state", None)
            if not callable(migrate):
                raise ValueError(
                    "Checkpoint uses a legacy state schema and this adapter has no migration"
                )
            migration_report = dict(
                migrate(env, checkpoint.get("pre_action_observation") or {}) or {}
            )
            if not migration_report.get("migration_applied"):
                raise ValueError(
                    "Legacy checkpoint migration was required but was not applied"
                )

        self.router.reset(seed)
        branch_id = "branch-" + canonical_hash(
            [
                intervention["source_run_id"],
                checkpoint["parent_episode_id"],
                fork_step,
                intervention["forced_action"],
                intervention["rationale"],
            ],
            20,
        )
        metadata = {
            "schema_version": BRANCH_SCHEMA_VERSION,
            "trajectory_type": "human_intervention_branch",
            "branch_id": branch_id,
            "parent_run_id": str(intervention["source_run_id"]),
            "parent_episode_id": str(intervention["parent_episode_id"]),
            "fork_step": fork_step,
            "original_action": checkpoint.get("original_action"),
            "forced_action": str(intervention["forced_action"]),
            "intervention_source": "human",
            "intervention_rationale": str(intervention["rationale"]),
            "prefix_inherited": True,
            "autonomous_suffix_start_step": fork_step + 1,
            "router_resume_mode": "observation_stateless_reset",
            "checkpoint_state_schema_migration": dict(migration_report),
        }
        return self._continue(
            row=row,
            manifest_path=manifest_path,
            adapter=adapter,
            env=env,
            seed=seed,
            fault_variant=str(checkpoint.get("private_execution_variant") or "clean"),
            history=list(checkpoint.get("prefix_history") or ()),
            raw_outputs=list(checkpoint.get("prefix_raw_outputs") or ()),
            steps=list(checkpoint.get("prefix_steps") or ()),
            calls_remaining=int(checkpoint["calls_remaining"]),
            forced_first_action=str(intervention["forced_action"]),
            branch_metadata=metadata,
            output_namespace=branch_id,
        )

    def _continue(
        self,
        *,
        row: Mapping[str, Any],
        manifest_path: str | Path,
        adapter,
        env,
        seed: int,
        fault_variant: str,
        history: Sequence[Mapping[str, Any]] = (),
        raw_outputs: Sequence[Mapping[str, Any]] = (),
        steps: Sequence[Mapping[str, Any]] = (),
        calls_remaining: int | None = None,
        forced_first_action: str | None = None,
        branch_metadata: Mapping[str, Any] | None = None,
        output_namespace: str,
    ) -> dict[str, Any]:
        profiles = {str(item["agent_id"]): dict(item) for item in row.get("agent_pool") or ()}
        candidate_actions = tuple(map(str, row.get("candidate_actions") or ()))
        if forced_first_action is not None and forced_first_action not in candidate_actions:
            raise ValueError("Forced action is outside the row candidate pool")
        history = [dict(item) for item in history]
        raw_outputs = [dict(item) for item in raw_outputs]
        steps = [dict(item) for item in steps]
        output_store = RawOutputStore(
            self.snapshot_store.root.parent / "outputs", namespace=output_namespace
        )
        calls_remaining = (
            self.max_calls if calls_remaining is None else int(calls_remaining)
        )
        intervention_pending = forced_first_action is not None
        terminal_reason = "CALL_LIMIT"
        while calls_remaining >= 0:
            pre_state = normalize_current_state(adapter.public_observation(env))
            observation = _router_observation(row=row, current_state=pre_state, history=history, raw_outputs=raw_outputs, calls_remaining=calls_remaining, memory_view=self.memory_view)
            private_phase = derive_private_phase(pre_state)
            if intervention_pending:
                valid = (str(forced_first_action),)
                annotation = {
                    "proposed_valid_actions": list(valid),
                    "review_status": "human_verified",
                    "label_scope": "intervention_step_only",
                }
                label_source = "human_intervention"
            elif self.action_judge is not None:
                annotation = self.action_judge.label_actions(decision_id=f"live-artifact:{row['decision_id']}:{len(steps):03d}", public_observation=observation, candidate_actions=candidate_actions)
                valid = tuple(map(str, annotation.get("proposed_valid_actions") or ()))
                label_source = "llm_pre_annotation_pending_human_review"
            else:
                valid = heuristic_valid_actions(stage=private_phase, candidate_actions=candidate_actions, profiles=profiles, initial_valid_actions=row.get("valid_actions") or ())
                annotation = {"proposed_valid_actions": list(valid), "review_status": "smoke_only"}
                label_source = "deterministic_smoke_heuristic"
            fingerprint = canonical_hash(pre_state, 20)
            snapshot = self.snapshot_store.create(env, public_state_fingerprint=fingerprint, budget=observation["budget"])
            if intervention_pending:
                selected_action = str(forced_first_action)
                router_metadata = {
                    "decision_source": "human_override",
                    "rationale": (branch_metadata or {}).get("intervention_rationale"),
                }
                decision_source = "human_override"
                intervention_pending = False
            else:
                selected = self.router.select(observation, candidate_actions, valid)
                selected_action = selected.action
                router_metadata = dict(selected.metadata)
                decision_source = "router"
            step = {
                "step_index": len(steps), "snapshot_id": snapshot.snapshot_id, "snapshot_dir": str(snapshot.snapshot_dir),
                "pre_state_fingerprint": fingerprint, "candidate_actions": list(candidate_actions), "valid_actions": list(valid),
                "valid_action_label_source": label_source, "action_annotation": annotation, "selected_action": selected_action,
                "selected_action_valid": selected_action in valid, "router_metadata": router_metadata, "pre_action_observation": observation,
                "decision_source": decision_source,
            }
            if selected_action == STOP_ACTION:
                terminal = bool(adapter.terminal_status(env).get("terminal"))
                status = "valid_stop" if terminal else "premature_stop"
                step["execution_status"] = status
                steps.append(step)
                history.append({"step": len(history) + 1, "agent_id": STOP_ACTION, "intent": "Terminate the workflow.", "execution_status": status, "state_delta": state_delta(pre_state, pre_state), "output_ref": None})
                terminal_reason = "VALID_STOP" if terminal else "PREMATURE_STOP"
                break
            if calls_remaining <= 0:
                step["execution_status"] = "call_limit"
                steps.append(step)
                break
            profile = profiles.get(selected_action)
            if profile is None:
                step["execution_status"] = "unknown_action"
                steps.append(step)
                terminal_reason = "UNKNOWN_ACTION"
                break
            precondition = adapter.public_preconditions(env, selected_action, profile)
            step["precondition"] = {"executable": precondition.executable, "reasons": list(precondition.reasons)}
            if not precondition.executable:
                actor_result = {"raw_output": "", "work_product": "", "raw_model_responses": [], "tool_events": [], "artifact_diff": {"added": [], "modified": [], "removed": []}, "transport_status": "pre_execution_rejection"}
            else:
                actor_result = self.actor.execute(env=env, adapter=adapter, action=selected_action, role_profile=profile, task=str(row["task"]))
            post_state = normalize_current_state(adapter.public_observation(env))
            delta = state_delta(pre_state, post_state)
            artifact_refs = []
            for item in delta.get("artifacts_added") or ():
                if item.get("artifact_id"):
                    artifact_refs.append(str(item["artifact_id"]))
            for item in delta.get("artifacts_updated") or ():
                if item.get("artifact_id"):
                    artifact_refs.append(str(item["artifact_id"]))
            output_record = output_store.write(step=len(raw_outputs) + 1, agent_id=selected_action, actor_result=actor_result, artifact_refs=artifact_refs)
            raw_outputs.append(output_record)
            execution_status = normalize_execution_status(actor_result, delta)
            failure_type = actor_result.get("failure_type")
            tool_results = [dict(item.get("tool_result") or {}) for item in actor_result.get("tool_events") or ()]
            if not failure_type:
                for result in reversed(tool_results):
                    candidate_failure = (result.get("output") or {}).get("failure_type")
                    if candidate_failure:
                        failure_type = str(candidate_failure)
                        break
            transport_status = str(actor_result.get("transport_status") or "success")
            tool_environment_failure = transport_status in {"transport_failure", "timeout"} or any(
                str(result.get("status")) in {"tool_failure", "timeout"}
                and str(((result.get("output") or {}).get("test_state") or {}).get("failure_kind", "")) == "environment"
                for result in tool_results
            )
            tool_trace = [
                {
                    "name": str((item.get("tool_call") or {}).get("name") or ""),
                    "arguments": dict((item.get("tool_call") or {}).get("arguments") or {}),
                    "status": str((item.get("tool_result") or {}).get("status") or ""),
                }
                for item in (actor_result.get("tool_events") or ())
            ]
            semantic_delta = "state_changed" if delta_has_change(delta) else "none"
            consecutive_same_state_attempts = 1
            if semantic_delta == "none" and history:
                previous = history[-1]
                if (
                    str(previous.get("agent_id") or "") == selected_action
                    and str(previous.get("pre_state_fingerprint") or "") == fingerprint
                    and str(previous.get("semantic_delta") or "none") == "none"
                ):
                    consecutive_same_state_attempts = int(
                        previous.get("consecutive_same_state_attempts") or 1
                    ) + 1
            event = {
                "step": len(history) + 1,
                "agent_id": selected_action,
                "intent": str(profile.get("role_goal") or profile.get("role_name") or ""),
                "status": _PUBLIC_EVENT_STATUS.get(
                    execution_status, "EXECUTED_NO_PROGRESS"
                ),
                "execution_status": execution_status,
                "failure_type": failure_type,
                "semantic_delta": semantic_delta,
                "pre_state_fingerprint": fingerprint,
                "attempt_signature": canonical_hash(
                    {
                        "state_fingerprint": fingerprint,
                        "agent_id": selected_action,
                        "tool_trace": tool_trace,
                    },
                    20,
                ),
                "consecutive_same_state_attempts": consecutive_same_state_attempts,
                "tools_used": [item["name"] for item in tool_trace if item["name"]],
                "new_evidence_refs": artifact_refs,
                "state_delta": delta,
                "output_ref": output_record["output_ref"],
            }
            history.append(event)
            step.update({"execution_status": execution_status, "failure_type": failure_type, "failure_class": "tool_environment" if tool_environment_failure else None, "executor_result": actor_result, "post_state": post_state, "history_event": event})
            steps.append(step)
            calls_remaining -= 1
        terminal = adapter.terminal_status(env)
        final_state = normalize_current_state(adapter.public_observation(env))
        post_task_judgment = None
        if self.post_task_judge is not None:
            post_task_judgment = self.post_task_judge.judge_post_task(
                task=str(row["task"]),
                task_input={"base_workflow_id": env.manifest.base_workflow_id, "domain": env.manifest.domain, "public_artifacts": [{"artifact_id": item.artifact_id, "path": item.path, "media_type": item.media_type, "sha256": item.sha256} for item in env.manifest.public_files]},
                final_state=final_state,
                trajectory_outputs=[{"agent_id": item.get("agent_id"), "raw_output": item.get("raw_output", ""), "artifact_refs": item.get("artifact_refs", [])} for item in raw_outputs],
                reference_answer=None,
                answer_aliases=(),
            )
        completed = bool(terminal.get("terminal"))
        provisional_origin = _failure_origin(steps, completed)
        labels_finalized = all(
            str((step.get("action_annotation") or {}).get("review_status", ""))
            in {"human_verified", "approved"}
            for step in steps
        )
        result = {
            "schema_version": SCHEMA_VERSION, "scenario_id": row.get("scenario_id"), "base_workflow_id": env.manifest.base_workflow_id,
            "domain": env.manifest.domain, "manifest_path": str(Path(manifest_path)), "memory_view": self.memory_view,
            "private_execution_variant": fault_variant, "steps": steps, "structured_history": history,
            "raw_output_records": [item.get("output_ref") for item in raw_outputs if item.get("output_ref")], "final_state": final_state,
            "terminal_reason": terminal_reason, "terminal_status": dict(terminal), "postcondition_validation": dict(adapter.validate_postcondition(env)),
            "post_task_judgment": post_task_judgment,
            "failure_origin": provisional_origin if labels_finalized else None,
            "provisional_failure_origin": provisional_origin,
            "failure_origin_status": "final" if labels_finalized else "pending_human_valid_action_review",
            "human_review_status": "pending",
            "seed": seed,
        }
        if branch_metadata:
            result["branch_metadata"] = dict(branch_metadata)
        return result
