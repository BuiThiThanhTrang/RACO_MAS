from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import canonical_hash, sha256_file

from ..base import (
    ArtifactDiff,
    EnvironmentHandle,
    PreconditionResult,
    ToolDefinition,
    ToolResult,
)


FINALIZER_TOKENS = ("final", "commit", "editor")
VALIDATOR_TOKENS = ("validat", "review", "audit", "verif")
RECOVERY_TOKENS = ("recover", "repair")
DIAGNOSIS_TOKENS = ("diagnos", "fault analyst")
DOWNSTREAM_TOKENS = ("workflow operator", "consumer")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ConcreteWorkflowAdapter:
    adapter_id = "concrete.v1"

    def materialize(
        self, manifest_path: str | Path, workspace: str | Path
    ) -> EnvironmentHandle:
        manifest_path = Path(manifest_path)
        pack_dir = manifest_path if manifest_path.is_dir() else manifest_path.parent
        manifest = load_manifest(pack_dir)
        if manifest.adapter_id != self.adapter_id:
            raise ValueError(
                f"Adapter mismatch: manifest={manifest.adapter_id}, runtime={self.adapter_id}"
            )
        workspace = Path(workspace)
        if workspace.exists():
            raise FileExistsError(f"Refusing to overwrite workspace: {workspace}")
        workspace.mkdir(parents=True)
        for item in (*manifest.public_files, *manifest.fixture_files):
            source = pack_dir / item.path
            destination = workspace / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        runtime = {
            "stage": "initial",
            "artifacts": {},
            "observable_issues": [],
            "terminal": False,
            "tool_events": [],
            "spent_cost": 0.0,
        }
        write_json(workspace / "_runtime/state.json", runtime)
        return EnvironmentHandle(manifest=manifest, pack_dir=pack_dir, workspace_dir=workspace)

    def _state(self, env: EnvironmentHandle) -> dict[str, Any]:
        return dict(read_json(env.workspace_dir / "_runtime/state.json"))

    def _save_state(self, env: EnvironmentHandle, state: Mapping[str, Any]) -> None:
        write_json(env.workspace_dir / "_runtime/state.json", state)

    def configure_fault(self, env: EnvironmentHandle, variant: str) -> None:
        """Configure a hidden execution variant without changing public inputs."""

        allowed = {
            "clean",
            "explicit_invalid",
            "silent_corruption",
            "retryable_failure",
            "conflicting_outputs",
        }
        if variant not in allowed:
            raise ValueError(f"Unknown fault variant: {variant}")
        state = self._state(env)
        state["_private_fault_variant"] = variant
        state["_private_fault_consumed"] = False
        self._save_state(env, state)

    def _reference(self, env: EnvironmentHandle) -> Mapping[str, Any]:
        candidates = [
            env.pack_dir / item.path
            for item in env.manifest.private_files
            if item.path.endswith("reference.json")
        ]
        return read_json(candidates[0]) if candidates else {}

    def public_observation(self, env: EnvironmentHandle) -> dict[str, Any]:
        state = self._state(env)
        runtime_artifacts = dict(state.get("artifacts") or {})
        artifacts = [
            {
                "artifact_id": item.artifact_id,
                "type": "task_input",
                "status": "available",
                "summary": "Executor-visible task input.",
                "content_ref": item.path,
                "evidence_refs": [],
            }
            for item in env.manifest.public_files
        ]
        for key, value in sorted(runtime_artifacts.items()):
            if not isinstance(value, Mapping):
                continue
            artifacts.append(
                {
                    "artifact_id": str(key).replace("_", "-"),
                    "type": str(value.get("artifact_type") or "workflow_artifact"),
                    "status": str(value.get("status") or "available"),
                    "summary": str(value.get("summary") or value.get("artifact_type") or key),
                    "content_ref": str(value.get("path") or ""),
                    "evidence_refs": list(value.get("evidence_refs") or ()),
                }
            )
        candidate_available = "candidate" in runtime_artifacts or "corrected_candidate" in runtime_artifacts
        validation = runtime_artifacts.get("validation") or {}
        validation_outcome = "pass" if validation.get("valid") is True else "fail" if validation.get("valid") is False else "not_run"
        downstream_ready = "downstream" in runtime_artifacts
        terminal = bool(state.get("terminal"))
        facts = [
            {"key": "task_input_available", "value": True, "evidence_refs": [item.artifact_id for item in env.manifest.public_files]},
            {"key": "candidate_available", "value": candidate_available, "evidence_refs": ["candidate"] if candidate_available else []},
            {"key": "validation_outcome", "value": validation_outcome, "evidence_refs": ["validation"] if validation_outcome != "not_run" else []},
            {"key": "downstream_artifact_ready", "value": downstream_ready, "evidence_refs": ["downstream"] if downstream_ready else []},
            {"key": "final_artifact_ready", "value": terminal, "evidence_refs": ["final"] if terminal else []},
        ]
        requirements = []
        if not candidate_available:
            requirements.append({"requirement_id": "produce-candidate", "description": "Produce a candidate artifact from the concrete task inputs.", "evidence_needed": "A concrete candidate artifact."})
        elif validation_outcome != "pass":
            requirements.append({"requirement_id": "validate-candidate", "description": "Validate the current candidate against the public contract.", "evidence_needed": "A fresh validation result."})
        elif not downstream_ready:
            requirements.append({"requirement_id": "execute-downstream", "description": "Execute the required downstream operation.", "evidence_needed": "Observable postcondition evidence."})
        elif not terminal:
            requirements.append({"requirement_id": "commit-final-artifact", "description": "Commit the supported artifact as the final result.", "evidence_needed": "A committed final artifact."})
        issues = [
            {
                "issue_id": f"issue-{index:03d}",
                "type": "observable_execution_issue",
                "summary": str(issue),
                "evidence_refs": [],
            }
            for index, issue in enumerate(state.get("observable_issues") or (), start=1)
        ]
        return {
            "task_status": "complete" if terminal else "active",
            "facts": facts,
            "artifacts": artifacts,
            "open_requirements": requirements,
            "issues": issues,
        }

    def public_preconditions(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> PreconditionResult:
        state = self._state(env)
        if state.get("terminal"):
            return PreconditionResult(False, ("workflow_already_terminal",))
        tools = self.tool_spec(env, action_id, role_profile)
        if not tools:
            return PreconditionResult(False, ("role_has_no_executable_tool",))
        return PreconditionResult(True, ("role_has_at_least_one_executable_tool",))

    def tool_spec(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> Sequence[ToolDefinition]:
        profile_text = " ".join(
            [
                str((role_profile or {}).get("role_name", "")),
                str((role_profile or {}).get("role_goal", "")),
                str((role_profile or {}).get("semantic_family", "")),
            ]
        ).casefold()
        inspect = ToolDefinition("inspect_inputs", "Read the concrete public inputs and contracts.")
        # Capability ownership is fixed by the role card. State determines
        # whether invoking that capability is useful, not whether it exists.
        if any(token in profile_text for token in FINALIZER_TOKENS):
            return (ToolDefinition("finalize", "Commit a validated downstream artifact."),)
        if any(token in profile_text for token in DIAGNOSIS_TOKENS):
            return (ToolDefinition("diagnose_failure", "Diagnose visible execution issues."),)
        if any(token in profile_text for token in RECOVERY_TOKENS):
            return (ToolDefinition("repair_candidate", "Repair or regenerate the candidate artifact."),)
        if any(token in profile_text for token in VALIDATOR_TOKENS):
            return (ToolDefinition("validate_candidate", "Validate the current candidate against the visible contract."),)
        if any(token in profile_text for token in DOWNSTREAM_TOKENS) or any(token in profile_text for token in ("synthes", "composer", "composition")):
            return (ToolDefinition("execute_downstream", "Apply the current candidate to the deterministic environment."),)
        return (inspect, ToolDefinition("produce_candidate", "Produce the domain-specific candidate artifact from public inputs."))

    def executor_context(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Pass current runtime artifacts to the executor, not to the router."""

        del action_id, role_profile
        state = self._state(env)
        context: dict[str, Any] = {}
        workspace = env.workspace_dir.resolve()
        for key, artifact in (state.get("artifacts") or {}).items():
            if not isinstance(artifact, Mapping) or not artifact.get("path"):
                continue
            path = (workspace / str(artifact["path"])).resolve()
            try:
                path.relative_to(workspace)
            except ValueError:
                continue
            if path.is_file() and path.suffix.casefold() == ".json":
                context[str(key)] = read_json(path)
        return context

    def execute_tool(
        self,
        env: EnvironmentHandle,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        arguments = dict(arguments or {})
        handlers = {
            "inspect_inputs": self._inspect_inputs,
            "produce_candidate": self._produce_candidate,
            "validate_candidate": self._validate_candidate,
            "execute_downstream": self._execute_downstream,
            "diagnose_failure": self._diagnose_failure,
            "repair_candidate": self._repair_candidate,
            "finalize": self._finalize,
        }
        # Domain adapters may expose additional bounded capabilities without
        # widening the contract of every adapter.  In particular, the live
        # code-debugging adapter owns read-only repository inspection tools;
        # they are only callable when that adapter advertises them in
        # ``tool_spec``.
        for name, method_name in (
            ("inspect_repository_file", "_inspect_repository_file"),
            ("search_repository", "_search_repository"),
        ):
            handler = getattr(self, method_name, None)
            if callable(handler):
                handlers[name] = handler
        if tool_name not in handlers:
            return ToolResult(tool_name, "rejected", {}, f"Unknown tool: {tool_name}", 0.0)
        result = handlers[tool_name](env, arguments)
        state = self._state(env)
        event = result.to_dict()
        event["event_index"] = len(state.get("tool_events") or ()) + 1
        state.setdefault("tool_events", []).append(event)
        state["spent_cost"] = float(state.get("spent_cost", 0.0)) + float(result.cost)
        self._save_state(env, state)
        return result

    def _inspect_inputs(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        previews: list[dict[str, Any]] = []
        for item in env.manifest.public_files:
            path = env.workspace_dir / item.path
            if item.media_type.startswith("text/") or item.media_type == "application/json":
                text = path.read_text(encoding="utf-8")
                preview = text[:4000]
            else:
                preview = f"binary artifact ({path.stat().st_size} bytes)"
            previews.append({"artifact_id": item.artifact_id, "path": item.path, "preview": preview})
        return ToolResult("inspect_inputs", "success", {"artifacts": previews}, "Concrete public inputs were inspected.", 0.25)

    def _produce_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        candidate = self.compute_candidate(env, arguments)
        variant = str(state.get("_private_fault_variant", "clean"))
        consumed = bool(state.get("_private_fault_consumed"))
        if not consumed and variant == "retryable_failure":
            state["_private_fault_consumed"] = True
            state["stage"] = "candidate_invalid"
            state["observable_issues"] = ["The attempted method returned no usable artifact."]
            self._save_state(env, state)
            return ToolResult(
                "produce_candidate",
                "tool_failure",
                {"retryable": True},
                "The concrete executor failed before producing a usable candidate.",
            )
        if not consumed and variant in {
            "explicit_invalid",
            "silent_corruption",
            "conflicting_outputs",
        }:
            candidate = {**dict(candidate), "_injected_conflict": variant}
            state["_private_fault_consumed"] = True
        path = env.workspace_dir / "_runtime/candidate.json"
        write_json(path, candidate)
        state["artifacts"]["candidate"] = {
            "artifact_type": "candidate_artifact",
            "path": "_runtime/candidate.json",
            "sha256": sha256_file(path),
            "status": "available",
        }
        if not consumed and variant in {"explicit_invalid", "conflicting_outputs"}:
            state["stage"] = "candidate_invalid"
            state["observable_issues"] = [
                "The candidate visibly conflicts with the supplied output contract."
            ]
            result_status = "invalid_output"
            result_message = "A candidate was produced with an observable contract conflict."
        else:
            # Silent corruption is intentionally observable only after validation.
            state["stage"] = "candidate"
            result_status = "success"
            result_message = "A concrete candidate artifact was produced."
        self._save_state(env, state)
        return ToolResult(
            "produce_candidate",
            result_status,
            {"candidate": candidate},
            result_message,
        )

    def _validate_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        candidate_path = env.workspace_dir / "_runtime/candidate.json"
        if not candidate_path.exists():
            return ToolResult("validate_candidate", "rejected", {}, "No candidate artifact is available.", 0.0)
        candidate = read_json(candidate_path)
        validation = self.validate_candidate(env, candidate)
        write_json(env.workspace_dir / "_runtime/validation.json", validation)
        state["artifacts"]["validation"] = {
            "artifact_type": "verification_record",
            "path": "_runtime/validation.json",
            "status": "available",
            "valid": bool(validation.get("valid")),
        }
        if validation.get("valid"):
            state["stage"] = "verified"
            state["observable_issues"] = []
            status = "success"
            message = "The candidate passed deterministic validation."
        else:
            state["stage"] = "candidate_invalid"
            state["observable_issues"] = list(validation.get("issues") or ("candidate_invalid",))
            status = "invalid_output"
            message = "The candidate failed deterministic validation."
        self._save_state(env, state)
        return ToolResult("validate_candidate", status, validation, message)

    def _execute_downstream(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        candidate_path = env.workspace_dir / "_runtime/candidate.json"
        if not candidate_path.exists():
            return ToolResult("execute_downstream", "rejected", {}, "No candidate artifact is available.", 0.0)
        candidate = read_json(candidate_path)
        downstream = self.apply_candidate(env, candidate)
        path = env.workspace_dir / "_runtime/downstream.json"
        write_json(path, downstream)
        state["artifacts"]["downstream"] = {
            "artifact_type": "downstream_artifact",
            "path": "_runtime/downstream.json",
            "sha256": sha256_file(path),
            "status": "available",
        }
        state["stage"] = "downstream"
        self._save_state(env, state)
        return ToolResult("execute_downstream", "success", {"downstream": downstream}, "The deterministic downstream operation completed.")

    def _diagnose_failure(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        issues = list(state.get("observable_issues") or ())
        if not issues:
            return ToolResult("diagnose_failure", "no_progress", {}, "No observable failure is available to diagnose.", 0.5)
        diagnosis = {"issues": issues, "recommended_scope": "regenerate_candidate_from_public_contract"}
        write_json(env.workspace_dir / "_runtime/diagnosis.json", diagnosis)
        state["artifacts"]["diagnosis"] = {"artifact_type": "fault_diagnosis", "path": "_runtime/diagnosis.json", "status": "available"}
        state["stage"] = "diagnosed"
        self._save_state(env, state)
        return ToolResult("diagnose_failure", "success", diagnosis, "Observable candidate failure was diagnosed.")

    def _repair_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        if "diagnosis" not in state.get("artifacts", {}):
            return ToolResult("repair_candidate", "rejected", {}, "A diagnosis is required before repair.", 0.0)
        result = self._produce_candidate(env, arguments)
        state = self._state(env)
        state["artifacts"]["corrected_candidate"] = dict(state["artifacts"]["candidate"])
        state["artifacts"]["corrected_candidate"]["artifact_type"] = "corrected_candidate_artifact"
        self._save_state(env, state)
        return ToolResult("repair_candidate", result.status, result.output, "The candidate was regenerated after diagnosis.", result.cost)

    def _finalize(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        downstream_path = env.workspace_dir / "_runtime/downstream.json"
        if not downstream_path.exists():
            return ToolResult("finalize", "rejected", {}, "A downstream artifact is required before final commitment.", 0.0)
        final = {"committed": True, "result": read_json(downstream_path), "base_workflow_id": env.manifest.base_workflow_id}
        path = env.workspace_dir / "_runtime/final.json"
        write_json(path, final)
        state["artifacts"]["final"] = {"artifact_type": "committed_final_artifact", "path": "_runtime/final.json", "sha256": sha256_file(path), "status": "committed"}
        state["stage"] = "complete"
        state["terminal"] = True
        self._save_state(env, state)
        return ToolResult("finalize", "success", final, "The final artifact was committed.", 0.0)

    def filesystem_hashes(self, env: EnvironmentHandle) -> dict[str, str]:
        return {
            path.relative_to(env.workspace_dir).as_posix(): sha256_file(path)
            for path in sorted(env.workspace_dir.rglob("*"))
            if path.is_file()
        }

    def artifact_diff(self, before: Mapping[str, str], after: Mapping[str, str]) -> ArtifactDiff:
        added = tuple({"path": path, "sha256": digest} for path, digest in sorted(after.items()) if path not in before)
        removed = tuple({"path": path, "sha256": digest} for path, digest in sorted(before.items()) if path not in after)
        modified = tuple({"path": path, "before_sha256": before[path], "after_sha256": digest} for path, digest in sorted(after.items()) if path in before and before[path] != digest)
        return ArtifactDiff(added=added, modified=modified, removed=removed)

    def validate_postcondition(self, env: EnvironmentHandle) -> Mapping[str, Any]:
        state = self._state(env)
        return {"valid": bool(state.get("terminal")), "stage": state.get("stage"), "issues": list(state.get("observable_issues") or ())}

    def terminal_status(self, env: EnvironmentHandle) -> Mapping[str, Any]:
        state = self._state(env)
        return {"terminal": bool(state.get("terminal")), "stage": state.get("stage"), "reason": "committed_final_artifact" if state.get("terminal") else "workflow_incomplete"}

    def compute_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def validate_candidate(self, env: EnvironmentHandle, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = self._reference(env)
        expected = reference.get("expected_candidate", reference.get("expected_result", reference.get("expected_fields", reference)))
        valid = candidate == expected
        return {"valid": valid, "issues": [] if valid else ["candidate_differs_from_reference"]}

    def apply_candidate(self, env: EnvironmentHandle, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"applied": True, "candidate": dict(candidate), "postcondition_observed": True}
