from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import hashlib
import json
from typing import Any, Mapping, Sequence

from routing_benchmark.v4.artifacts.schema import sha256_file

from ..base import EnvironmentHandle, PreconditionResult, ToolDefinition, ToolResult
from .code_debugging import CodeDebuggingAdapter
from .common import read_json, write_json


class ExternalCodeDebuggingAdapter(CodeDebuggingAdapter):
    """Execute source-grounded debugging packets against a real checkout.

    The router observes only the compact test state. Raw test output and source
    files are returned to the selected executor through tool results.
    """

    adapter_id = "code_debugging.external.v1"
    python_environment_variable = "ROUTINGBENCH_CODE_PYTHON"
    max_output_chars = 20_000
    max_repository_file_bytes = 1_000_000
    max_repository_read_lines = 200
    max_repository_search_results = 20
    max_repository_search_files = 5_000
    excluded_repository_parts = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            ".tox",
            ".venv",
            "venv",
            "node_modules",
            "site-packages",
            "__pycache__",
            # QuixBugs ships reference implementations beside the buggy
            # programs.  They are evaluator artifacts, not executor-visible
            # repository context.
            "correct_python_programs",
            "correct_java_programs",
        }
    )

    def __init__(self) -> None:
        self._runtime_defaults: dict[str, Any] = {}

    def configure_runtime(self, profile: Mapping[str, Any] | None) -> None:
        """Attach evaluator-side runtime paths without exposing them to the router."""

        self._runtime_defaults = dict(profile or {})

    def public_preconditions(self, env, action_id, role_profile=None):
        state=self._state(env)
        if action_id=='Patch Author' and not state.get('checkout_ready'):
            return PreconditionResult(False,('No pinned repository checkout is available',))
        if action_id=='Repository Test Reproducer' and state.get('artifacts',{}).get('candidate',{}).get('status')=='applied':
            return PreconditionResult(False,('Baseline reproduction cannot reset or relabel a checkout with an applied candidate',))
        return super().public_preconditions(env,action_id,role_profile)

    def _cumulative_patch(self, env):
        repository=self._repository(env)
        names=self._run(['git','-c','core.quotepath=false','diff','--name-only','HEAD'],cwd=repository).stdout.splitlines()
        names=[name for name in names if self.patch_path_allowed(name)]
        text=self._run(['git','diff','--binary','HEAD','--',*names],cwd=repository).stdout if names else ''
        for name in self._run(['git','ls-files','--others','--exclude-standard'],cwd=repository).stdout.splitlines():
            if self.patch_path_allowed(name):
                text+=self._run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=repository).stdout
        return text

    def _preserve_rejected_attempt(self, env, patch_path, message):
        state=self._state(env)
        state['observable_issues']=[message]
        state.setdefault('artifacts',{})['rejected_candidate']={'artifact_type':'candidate_patch',
            'path':patch_path.relative_to(env.workspace_dir).as_posix(),'sha256':sha256_file(patch_path),
            'status':'invalid','summary':message}
        self._save_state(env,state)

    def evidence_version(self, env):
        repository=self._repository(env)
        if not repository.exists(): return None
        diff=self._run(['git','diff','--binary','HEAD'],cwd=repository)
        new_files={name:sha256_file(repository/name) for name in self._run(['git','ls-files','--others','--exclude-standard'],cwd=repository).stdout.splitlines()
                   if self.patch_path_allowed(name) and (repository/name).is_file()}
        public={item.path:sha256_file(env.workspace_dir/item.path) for item in env.manifest.public_files}
        return hashlib.sha256(json.dumps({'diff':diff.stdout,'new_files':new_files,'public':public,'runtime':self._runtime_defaults},sort_keys=True).encode()).hexdigest()

    def terminal_status(self, env):
        state=self._state(env)
        evidence=state.get('validated_version')
        terminal=bool(state.get('candidate_test_outcome')=='pass' and evidence and evidence==self.evidence_version(env))
        return {'terminal':terminal,'stage':state.get('stage'),'reason':'fresh_verified_patch' if terminal else 'workflow_incomplete'}

    def validate_postcondition(self, env):
        return {'valid':self.terminal_status(env)['terminal']}

    @staticmethod
    def patch_path_allowed(name):
        path=Path(name)
        return (not path.is_absolute() and '..' not in path.parts and path.suffix in ('.py','.pyi')
                and not any(part.lower() in ('tests','test','testing','.git','correct_python_programs','correct_java_programs') for part in path.parts)
                and not path.name.startswith(('test_','conftest'))
                and path.name not in ('setup.py','noxfile.py'))

    def _source_excerpts_from_test_result(
        self,
        env: EnvironmentHandle,
        result: Mapping[str, Any],
        *,
        context_lines: int = 12,
        max_files: int = 4,
    ) -> list[dict[str, Any]]:
        """Expose traceback-grounded checkout excerpts to the executor only.

        The paths and line numbers come from empirical test output, not private
        gold patches.  Resolving them under the pinned repository prevents path
        traversal and gives a patch author the same concrete source context a
        human debugger would inspect after reproducing a failure.
        """

        repository = self._repository(env).resolve()
        raw_output = str(result.get("raw_output") or "")
        matches = re.findall(r'File\s+"([^"]+)",\s+line\s+(\d+)', raw_output)
        excerpts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_path, raw_line in matches:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = repository / candidate
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(repository)
            except (OSError, ValueError):
                continue
            relative_name = relative.as_posix()
            if relative_name in seen or not resolved.is_file():
                continue
            try:
                lines = resolved.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            line_number = max(1, int(raw_line))
            start = max(1, line_number - context_lines)
            end = min(len(lines), line_number + context_lines)
            numbered = "\n".join(
                f"{index:>6}: {lines[index - 1]}"
                for index in range(start, end + 1)
            )
            excerpts.append(
                {
                    "path": relative_name,
                    "traceback_line": line_number,
                    "line_start": start,
                    "line_end": end,
                    "content": numbered,
                    "provenance": "empirical_traceback",
                }
            )
            seen.add(relative_name)
            if len(excerpts) >= max_files:
                break
        return excerpts

    def migrate_checkpoint_state(
        self,
        env: EnvironmentHandle,
        legacy_observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Migrate a pre-4.3 code checkpoint inside an isolated branch workspace.

        Older empirical runs persisted ``stage`` and one generic ``test_state``.
        Current adapters use explicit baseline, patch, and candidate-test outcomes.
        The source snapshot is never modified; this method only updates the restored
        branch after snapshot equivalence has already been verified.
        """

        state = self._state(env)
        if "baseline_test_outcome" in state:
            return {"migration_applied": False, "reason": "current_state_schema"}
        stage = str(state.get("stage") or "initial")
        test_state = dict(state.get("test_state") or {})
        observed_outcome = str(test_state.get("outcome") or "not_run")
        candidate_exists = bool(
            (state.get("artifacts") or {}).get("candidate")
            or (env.workspace_dir / "_runtime/candidate.patch").is_file()
        )
        if stage in {"verified", "downstream", "complete"}:
            candidate_outcome = "pass"
        elif stage == "candidate_invalid" and candidate_exists:
            candidate_outcome = "fail"
        else:
            candidate_outcome = "not_run"
        baseline_outcome = (
            observed_outcome
            if stage
            in {
                "baseline_reproduced",
                "diagnosed",
                "candidate",
                "candidate_invalid",
                "verified",
                "downstream",
                "complete",
            }
            and observed_outcome in {"pass", "fail", "error"}
            else "not_run"
        )
        if candidate_outcome in {"pass", "fail"}:
            baseline_outcome = "fail"
        state["checkout_ready"] = bool(state.get("checkout_ready"))
        state["baseline_test_outcome"] = baseline_outcome
        state["candidate_patch_outcome"] = (
            "applied" if candidate_exists else "absent"
        )
        state["candidate_test_outcome"] = candidate_outcome
        state["final_artifact_ready"] = stage in {"downstream", "complete"}

        artifacts = state.setdefault("artifacts", {})
        old_result = env.workspace_dir / "_runtime/test_result.json"
        baseline_result = env.workspace_dir / "_runtime/baseline_test_result.json"
        if old_result.is_file() and not baseline_result.exists():
            shutil.copy2(old_result, baseline_result)
        if baseline_result.is_file() and baseline_outcome != "not_run":
            artifacts["baseline_test_result"] = {
                "artifact_type": "verification_record",
                "path": "_runtime/baseline_test_result.json",
                "status": "available" if baseline_outcome != "error" else "error",
                "passed": baseline_outcome == "pass",
                "summary": f"Migrated fresh baseline result: {baseline_outcome}.",
                "evidence_refs": [],
            }

        if stage == "diagnosed" and "diagnosis" not in artifacts:
            recent = list(legacy_observation.get("recent_events") or ())
            diagnosis_text = next(
                (
                    str(item.get("effect") or item.get("raw_output") or "").strip()
                    for item in reversed(recent)
                    if "diagnos" in str(item.get("agent_id") or "").casefold()
                ),
                "Legacy trajectory recorded a completed diagnosis.",
            )
            diagnosis = {
                "analysis": diagnosis_text,
                "evidence_refs": ["baseline-test-result"],
                "source_outcome": baseline_outcome,
                "migration_source": "legacy_checkpoint",
            }
            write_json(env.workspace_dir / "_runtime/diagnosis.json", diagnosis)
            artifacts["diagnosis"] = {
                "artifact_type": "diagnosis",
                "path": "_runtime/diagnosis.json",
                "status": "candidate",
                "summary": "Diagnosis migrated from the legacy public trajectory.",
                "evidence_refs": ["baseline-test-result"],
            }
        self._save_state(env, state)
        return {
            "migration_applied": True,
            "source_stage": stage,
            "baseline_test_outcome": baseline_outcome,
            "candidate_patch_outcome": state["candidate_patch_outcome"],
            "candidate_test_outcome": candidate_outcome,
        }

    def materialize(self, manifest_path: str | Path, workspace: str | Path) -> EnvironmentHandle:
        env = super().materialize(Path(manifest_path).resolve(), Path(workspace).resolve())
        state = self._state(env)
        state["checkout_ready"] = False
        state["baseline_test_outcome"] = "not_run"
        state["candidate_patch_outcome"] = "absent"
        state["candidate_test_outcome"] = "not_run"
        state["final_artifact_ready"] = False
        self._save_state(env, state)
        return env

    def _contract(self, env: EnvironmentHandle) -> Mapping[str, Any]:
        return read_json(env.workspace_dir / "public/execution_contract.json")

    def public_observation(self, env: EnvironmentHandle) -> dict[str, Any]:
        state = self._state(env)
        artifacts: list[dict[str, Any]] = []
        for item in env.manifest.public_files:
            if item.path == "public/execution_contract.json" or "/tests/" in item.path:
                artifact_type = "test_contract"
            elif "/repo/" in item.path:
                artifact_type = "source_snapshot"
            else:
                artifact_type = "source_snapshot"
            artifacts.append(
                {
                    "artifact_id": item.artifact_id,
                    "type": artifact_type,
                    "status": "available",
                    "summary": "Executor-visible task input.",
                    "content_ref": item.path,
                    "evidence_refs": [],
                }
            )

        runtime_artifacts = dict(state.get("artifacts") or {})
        artifact_specs = {
            "repository_checkout": ("repository-checkout", "repository_checkout", "available"),
            "baseline_test_result": ("baseline-test-result", "test_result", "available"),
            "diagnosis": ("diagnosis", "diagnosis", "candidate"),
            "candidate": ("candidate-patch", "candidate_patch", "candidate"),
            "candidate_test_result": ("candidate-test-result", "test_result", "available"),
            "downstream": ("verified-patch", "verified_patch", "verified"),
            "final": ("final-deliverable", "final_deliverable", "finalized"),
        }
        for key, (artifact_id, artifact_type, default_status) in artifact_specs.items():
            value = runtime_artifacts.get(key)
            if not isinstance(value, Mapping):
                continue
            status = default_status
            if key == "candidate":
                patch_outcome = str(state.get("candidate_patch_outcome", "absent"))
                if patch_outcome == "rejected":
                    status = "invalid"
                elif str(state.get("candidate_test_outcome")) == "fail":
                    status = "invalid"
                elif str(state.get("candidate_test_outcome")) == "pass":
                    status = "verified"
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "type": artifact_type,
                    "status": status,
                    "summary": str(value.get("summary") or value.get("artifact_type") or artifact_type),
                    "content_ref": str(value.get("path") or ""),
                    "evidence_refs": list(value.get("evidence_refs") or ()),
                }
            )

        checkout_ready = bool(state.get("checkout_ready"))
        baseline_outcome = str(state.get("baseline_test_outcome", "not_run"))
        patch_outcome = str(state.get("candidate_patch_outcome", "absent"))
        candidate_outcome = str(state.get("candidate_test_outcome", "not_run"))
        final_ready = bool(state.get("final_artifact_ready"))
        terminal = self.terminal_status(env)['terminal']
        facts = [
            {
                "key": "checkout_ready",
                "value": checkout_ready,
                "evidence_refs": ["repository-checkout"] if checkout_ready else ["public-execution_contract.json"],
            },
            {
                "key": "baseline_test_outcome",
                "value": baseline_outcome,
                "evidence_refs": ["baseline-test-result"] if baseline_outcome != "not_run" else ["public-execution_contract.json"],
            },
            {
                "key": "candidate_patch_outcome",
                "value": patch_outcome,
                "evidence_refs": ["candidate-patch"] if patch_outcome != "absent" else ["public-execution_contract.json"],
            },
            {
                "key": "candidate_test_outcome",
                "value": candidate_outcome,
                "evidence_refs": ["candidate-test-result"] if candidate_outcome != "not_run" else ["public-execution_contract.json"],
            },
            {
                "key": "final_artifact_ready",
                "value": final_ready,
                "evidence_refs": ["verified-patch"] if final_ready else ["public-execution_contract.json"],
            },
        ]

        requirements: list[dict[str, Any]] = []
        if baseline_outcome != "fail":
            requirements.append(
                {
                    "requirement_id": "reproduce-failure",
                    "description": "Reproduce the reported defect using the supplied regression contract.",
                    "evidence_needed": "A fresh failing baseline test result.",
                }
            )
        if patch_outcome != "applied" and candidate_outcome != "pass":
            requirements.append(
                {
                    "requirement_id": "produce-candidate",
                    "description": "Produce an applicable candidate repair from the concrete inputs.",
                    "evidence_needed": "An executor-authored patch accepted by the patch validator.",
                }
            )
        if patch_outcome == "applied" and candidate_outcome != "pass":
            requirements.append(
                {
                    "requirement_id": "validate-candidate",
                    "description": "Validate the candidate using the supplied regression contract.",
                    "evidence_needed": "A fresh candidate test result.",
                }
            )
        if candidate_outcome == "pass" and not terminal:
            requirements.append(
                {
                    "requirement_id": "produce-final-artifact",
                    "description": "Produce and commit the final deliverable from verified evidence.",
                    "evidence_needed": "A finalized deliverable referencing the verified patch and fresh test result.",
                }
            )

        issues: list[dict[str, Any]] = []
        if baseline_outcome == "error" or candidate_outcome == "error":
            issues.append(
                {
                    "issue_id": "issue-environment-setup-failed",
                    "type": "environment_setup_failed",
                    "summary": "The concrete test environment failed to execute successfully.",
                    "evidence_refs": ["candidate-test-result" if candidate_outcome == "error" else "baseline-test-result"],
                }
            )
        if baseline_outcome == "pass" and patch_outcome == "absent":
            issues.append(
                {
                    "issue_id": "issue-failure-not-reproduced",
                    "type": "failure_not_reproduced",
                    "summary": "The supplied regression did not reproduce the reported defect.",
                    "evidence_refs": ["baseline-test-result"],
                }
            )
        if patch_outcome == "rejected":
            issues.append(
                {
                    "issue_id": "issue-patch-not-applicable",
                    "type": "patch_not_applicable",
                    "summary": "The proposed candidate patch could not be applied.",
                    "evidence_refs": ["candidate-patch"],
                }
            )
        if (baseline_outcome == "fail" and candidate_outcome != "pass") or candidate_outcome == "fail":
            evidence = "candidate-test-result" if candidate_outcome == "fail" else "baseline-test-result"
            issues.append(
                {
                    "issue_id": "issue-regression-failed",
                    "type": "regression_failed",
                    "summary": "The supplied implementation or current candidate fails the regression contract.",
                    "evidence_refs": [evidence],
                }
            )

        if terminal:
            task_status = "complete"
        elif final_ready:
            task_status = "ready_to_finish"
        elif baseline_outcome == "error" or candidate_outcome == "error":
            task_status = "blocked"
        else:
            task_status = "active"
        return {
            'execution_contract':self._contract(env),
            'patch_scope':{'allowed':'Python implementation files (.py/.pyi); multiple files permitted',
                           'protected':'tests/test/testing directories, test_*, conftest*, setup.py, noxfile.py, reference implementations; paths must stay in checkout'},
            'terminal_contract':'fresh passing candidate validation; promotion and finalization are optional',
            "task_status": task_status,
            "facts": facts,
            "artifacts": artifacts,
            "open_requirements": requirements,
            "issues": issues,
        }

    def tool_spec(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> Sequence[ToolDefinition]:
        del env
        profile = dict(role_profile or {})
        profile_text = " ".join(
            str(profile.get(key, ""))
            for key in ("role_name", "role_goal", "semantic_family")
        ).casefold()
        semantic_family = str(profile.get("semantic_family") or "").strip().casefold()
        # Tools are capability affordances of a role. They are deliberately not
        # removed by workflow state; the tool/validator reports whether a call
        # was useful, rejected, or failed in the current concrete environment.
        is_reproducer = semantic_family == "reproduction" or "reproducer" in profile_text
        is_repair = semantic_family == "repair" or any(
            token in profile_text for token in ("patch author", "repair")
        )
        is_diagnostician = (
            semantic_family == "diagnosis"
            or ("diagnos" in profile_text and not is_repair)
        )
        is_verifier = semantic_family == "verification" or any(
            token in profile_text for token in ("regression verifier", "validat")
        )
        is_consumer = semantic_family in {"consumption", "downstream"} or any(
            token in profile_text
            for token in ("verified patch consumer", "workflow operator", "consumer")
        )
        is_finalizer = semantic_family in {"finalization", "commit"} or any(
            token in profile_text for token in ("patch finalizer", "finalizer", "commit")
        )
        # Canonical roles own fixed tools. Incidental words in prose (e.g.
        # "repair constraints" in a diagnosis card) cannot grant patch access.
        canonical=('Repository Test Reproducer','Patch Author','Failure Diagnostician',
                   'Regression Verifier','Verified Patch Consumer','Patch Finalizer')
        if action_id in canonical:
            is_reproducer,is_repair,is_diagnostician,is_verifier,is_consumer,is_finalizer=tuple(action_id==name for name in canonical)

        inspect = ToolDefinition("inspect_inputs", "Read the concrete public inputs and contracts.")
        inspect_repository_file = ToolDefinition(
            "inspect_repository_file",
            (
                "Read an exact, line-numbered excerpt from one text file in the pinned "
                "buggy checkout. Paths are repository-relative; private evaluator artifacts "
                "and files outside the checkout are inaccessible."
            ),
            {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        )
        search_repository = ToolDefinition(
            "search_repository",
            (
                "Search for a literal string in bounded text files inside the pinned buggy "
                "checkout. Returns repository-relative paths and matching line previews only."
            ),
            {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "file_glob": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.max_repository_search_results,
                    },
                },
                "additionalProperties": False,
            },
        )
        if is_reproducer:
            return (
                ToolDefinition(
                    "produce_candidate",
                    "Materialize the pinned checkout, install the executor-visible reproducer, and run it once.",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                ),
            )
        if is_repair:
            return (
                inspect,
                inspect_repository_file,
                search_repository,
                ToolDefinition(
                    "repair_candidate",
                    "Apply the executor-authored unified diff to the pinned checkout.",
                    {
                        "type": "object",
                        "required": ["patch"],
                        "properties": {"patch": {"type": "string"}},
                    },
                ),
            )
        if is_diagnostician:
            return (
                inspect,
                inspect_repository_file,
                search_repository,
                ToolDefinition(
                    "diagnose_failure",
                    "Record an executor-authored diagnosis grounded in the supplied concrete test output.",
                    {
                        "type": "object",
                        "required": ["analysis"],
                        "properties": {"analysis": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ),
            )
        if is_verifier:
            return (
                ToolDefinition(
                    "validate_candidate",
                    "Run the executor-visible regression contract against the applied candidate patch.",
                ),
            )
        if is_consumer:
            return (
                ToolDefinition(
                    "execute_downstream",
                    "Promote the fresh passing patch into the downstream workflow artifact.",
                ),
            )
        if is_finalizer:
            return (ToolDefinition("finalize", "Commit the validated downstream patch as the terminal result."),)
        return ()

    def executor_context(
        self,
        env: EnvironmentHandle,
        action_id: str,
        role_profile: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return only role-relevant empirical artifacts, never private oracle data."""

        del action_id
        profile_text = " ".join(
            str((role_profile or {}).get(key, ""))
            for key in ("role_name", "role_goal", "semantic_family")
        ).casefold()
        context: dict[str, Any] = {'state':self.public_observation(env)}
        # New role invocations must see prior tool feedback, including a rejected
        # patch. This is shared evidence, never a preferred next-role hint.
        context['recent_tool_feedback']=[{key:event.get(key) for key in ('tool_name','status','observable_message','output')}
            for event in self._state(env).get('tool_events',[])[-3:]]
        if "diagnos" in profile_text or any(token in profile_text for token in ("patch author", "repair")):
            for name in ("candidate_test_result.json", "baseline_test_result.json"):
                path = env.workspace_dir / "_runtime" / name
                if path.exists():
                    value = read_json(path)
                    context["latest_test_result"] = {
                        "outcome": value.get("outcome"),
                        "return_code": value.get("return_code"),
                        "raw_output": value.get("raw_output", ""),
                        "evidence_ref": "candidate-test-result" if name.startswith("candidate") else "baseline-test-result",
                    }
                    excerpts = self._source_excerpts_from_test_result(env, value)
                    if excerpts:
                        context["traceback_source_excerpts"] = excerpts
                    break
        if any(token in profile_text for token in ("patch author", "repair")):
            diagnosis_path = env.workspace_dir / "_runtime/diagnosis.json"
            if diagnosis_path.exists():
                context["latest_diagnosis"] = read_json(diagnosis_path)
        return context

    def _inspect_inputs(
        self,
        env: EnvironmentHandle,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Inspect public packet inputs plus source grounded by test traceback."""

        base = super()._inspect_inputs(env, arguments)
        output = dict(base.output)
        result_path = env.workspace_dir / "_runtime/candidate_test_result.json"
        if not result_path.exists():
            result_path = env.workspace_dir / "_runtime/baseline_test_result.json"
        excerpts = (
            self._source_excerpts_from_test_result(env, read_json(result_path))
            if result_path.exists()
            else []
        )
        if excerpts:
            output["traceback_source_excerpts"] = excerpts
        return ToolResult(
            "inspect_inputs",
            base.status,
            output,
            (
                "Concrete public inputs and traceback-grounded repository excerpts were inspected."
                if excerpts
                else base.observable_message
            ),
            base.cost,
        )

    def _repository(self, env: EnvironmentHandle) -> Path:
        return env.workspace_dir / "_external/repository"

    def _resolve_repository_file(
        self,
        env: EnvironmentHandle,
        raw_path: Any,
    ) -> tuple[Path | None, str | None, str | None]:
        """Resolve a bounded executor path without exposing evaluator files."""

        repository = self._repository(env).resolve()
        if not repository.is_dir():
            return None, None, "The pinned repository checkout is not available."
        value = str(raw_path or "").strip().replace("\\", "/")
        # QuixBugs exposes a deliberately small single-file public view while
        # execution happens against the pinned upstream checkout.  Accept the
        # public alias (and the basename models commonly infer) but resolve it
        # only to the declared buggy source.  Never resolve the paired correct
        # implementation.
        try:
            contract = self._contract(env)
        except (OSError, ValueError, KeyError, TypeError):
            contract = {}
        if str(contract.get("test_interface") or "") == "standalone_jsonl_runner":
            provenance_path = env.workspace_dir / "fixtures/upstream_provenance.json"
            if provenance_path.is_file():
                provenance = read_json(provenance_path)
                buggy_source = str(
                    (provenance.get("source_paths") or {}).get("buggy") or ""
                ).replace("\\", "/")
                buggy_name = Path(buggy_source).name
                if buggy_source and value in {buggy_name, f"public/repo/{buggy_name}"}:
                    value = buggy_source
        candidate = Path(value)
        if (
            not value
            or candidate.is_absolute()
            or re.match(r"^[A-Za-z]:", value)
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            return None, None, "A safe repository-relative file path is required."
        try:
            resolved = (repository / candidate).resolve()
            relative = resolved.relative_to(repository)
        except (OSError, ValueError):
            return None, None, "The requested path is outside the pinned repository."
        if any(part.casefold() in self.excluded_repository_parts for part in relative.parts):
            return None, None, "The requested repository path is not executor-visible."
        if not resolved.is_file():
            return None, relative.as_posix(), "The requested repository file does not exist."
        try:
            if resolved.stat().st_size > self.max_repository_file_bytes:
                return None, relative.as_posix(), "The requested repository file exceeds the read limit."
        except OSError:
            return None, relative.as_posix(), "The requested repository file could not be inspected."
        return resolved, relative.as_posix(), None

    @staticmethod
    def _read_text_file(path: Path) -> tuple[str | None, str | None]:
        try:
            payload = path.read_bytes()
        except OSError as error:
            return None, f"{type(error).__name__}: {error}"
        if b"\x00" in payload:
            return None, "Binary files are not executor-visible through repository inspection."
        try:
            return payload.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, "The requested file is not UTF-8 text."

    def _inspect_repository_file(
        self,
        env: EnvironmentHandle,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        path, relative, error = self._resolve_repository_file(env, arguments.get("path"))
        if error or path is None or relative is None:
            status = "no_progress" if "checkout" in str(error).casefold() else "rejected"
            return ToolResult(
                "inspect_repository_file",
                status,
                {"path": str(arguments.get("path") or "")},
                str(error),
                0.0,
            )
        text, read_error = self._read_text_file(path)
        if read_error or text is None:
            return ToolResult(
                "inspect_repository_file",
                "rejected",
                {"path": relative},
                str(read_error),
                0.0,
            )
        lines = text.splitlines()
        try:
            start = max(1, int(arguments.get("start_line", 1)))
            requested_end = int(
                arguments.get(
                    "end_line",
                    start + self.max_repository_read_lines - 1,
                )
            )
        except (TypeError, ValueError):
            return ToolResult(
                "inspect_repository_file",
                "rejected",
                {"path": relative},
                "start_line and end_line must be integers.",
                0.0,
            )
        end = max(start, requested_end)
        end = min(end, start + self.max_repository_read_lines - 1, len(lines))
        if start > len(lines) and lines:
            return ToolResult(
                "inspect_repository_file",
                "no_progress",
                {"path": relative, "total_lines": len(lines)},
                "The requested start line is beyond the end of the file.",
                0.0,
            )
        content = "\n".join(
            f"{index:>6}: {lines[index - 1]}"
            for index in range(start, end + 1)
        )
        return ToolResult(
            "inspect_repository_file",
            "success",
            {
                "path": relative,
                "line_start": start,
                "line_end": end,
                "total_lines": len(lines),
                "content": content,
                "sha256": sha256_file(path),
                "truncated": end < len(lines),
                "provenance": "pinned_buggy_checkout",
            },
            f"Read exact source lines {start}-{end} from {relative}.",
            0.25,
        )

    def _search_repository(
        self,
        env: EnvironmentHandle,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        repository = self._repository(env).resolve()
        if not repository.is_dir():
            return ToolResult(
                "search_repository",
                "no_progress",
                {},
                "The pinned repository checkout is not available.",
                0.0,
            )
        query = str(arguments.get("query") or "").strip()
        if not query or len(query) > 200 or "\x00" in query:
            return ToolResult(
                "search_repository",
                "rejected",
                {},
                "A non-empty literal query of at most 200 characters is required.",
                0.0,
            )
        file_glob = str(arguments.get("file_glob") or "").strip()
        try:
            limit = int(
                arguments.get("max_results", self.max_repository_search_results)
            )
        except (TypeError, ValueError):
            limit = self.max_repository_search_results
        limit = max(1, min(limit, self.max_repository_search_results))
        needle = query.casefold()
        matches: list[dict[str, Any]] = []
        scanned = 0
        stopped_early = False
        for root, directories, filenames in os.walk(repository):
            directories[:] = sorted(
                name
                for name in directories
                if name.casefold() not in self.excluded_repository_parts
            )
            for filename in sorted(filenames):
                path = Path(root) / filename
                try:
                    relative = path.resolve().relative_to(repository)
                except (OSError, ValueError):
                    continue
                if any(
                    part.casefold() in self.excluded_repository_parts
                    for part in relative.parts
                ):
                    continue
                relative_name = relative.as_posix()
                if file_glob and not Path(relative_name).match(file_glob):
                    continue
                scanned += 1
                if scanned > self.max_repository_search_files:
                    stopped_early = True
                    break
                try:
                    if path.stat().st_size > self.max_repository_file_bytes:
                        continue
                except OSError:
                    continue
                text, _ = self._read_text_file(path)
                if text is None:
                    continue
                for line_number, line in enumerate(text.splitlines(), 1):
                    column = line.casefold().find(needle)
                    if column < 0:
                        continue
                    matches.append(
                        {
                            "path": relative_name,
                            "line": line_number,
                            "column": column + 1,
                            "content": line[:500],
                            "provenance": "pinned_buggy_checkout",
                        }
                    )
                    if len(matches) >= limit:
                        stopped_early = True
                        break
                if len(matches) >= limit:
                    break
            if stopped_early:
                break
        status = "success" if matches else "no_progress"
        return ToolResult(
            "search_repository",
            status,
            {
                "query": query,
                "file_glob": file_glob or None,
                "matches": matches,
                "match_count": len(matches),
                "scanned_files": min(scanned, self.max_repository_search_files),
                "truncated": stopped_early,
                "provenance": "pinned_buggy_checkout",
            },
            (
                f"Found {len(matches)} bounded repository match(es) for the literal query."
                if matches
                else "No repository text matched the literal query within the search bounds."
            ),
            0.25,
        )

    @staticmethod
    def _run(
        arguments: list[str],
        *,
        cwd: Path,
        timeout: int = 300,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        process_environment = os.environ.copy()
        process_environment.update(dict(environment or {}))
        return subprocess.run(
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=process_environment,
        )

    @staticmethod
    def _revision(contract: Mapping[str, Any]) -> str:
        revision = contract.get("revision") or contract.get("source_revision")
        if not revision:
            raise RuntimeError("The public execution contract has no pinned revision")
        return str(revision)

    def _materialize_checkout(self, env: EnvironmentHandle) -> Mapping[str, Any]:
        contract = self._contract(env)
        repository = self._repository(env)
        revision = self._revision(contract)
        marker_path = env.workspace_dir / "_runtime/checkout_materialization.json"
        state = self._state(env)

        # A tool call can finish materializing the checkout and then be rejected
        # by the actor protocol (for example, a tool-only response with no final
        # message).  Retrying by deleting the freshly used Git directory is
        # unreliable on Windows because pack files can remain locked briefly.
        # Reuse only a checkout that this adapter marked as complete, is still at
        # the pinned revision. Existing source edits are never reset by reuse.
        if (
            repository.is_dir()
            and marker_path.is_file()
        ):
            marker = read_json(marker_path)
            head = self._run(["git", "rev-parse", "HEAD"], cwd=repository)
            resolved_head = head.stdout.strip() if head.returncode == 0 else ""
            if (
                str(marker.get("requested_revision", "")) == revision
                and str(marker.get("resolved_revision", "")) == resolved_head
                and resolved_head
            ):
                return {
                    "repository": str(repository),
                    "revision": resolved_head,
                    "reused": True,
                }

        if repository.exists():
            if not repository.resolve().is_relative_to(env.workspace_dir.resolve()):
                raise ValueError("Checkout target escaped the task workspace")
            shutil.rmtree(repository)
        repository.parent.mkdir(parents=True, exist_ok=True)
        clone = self._run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                str(self._runtime_defaults.get("repository_mirror") or contract["repository_url"]),
                str(repository),
            ],
            cwd=env.workspace_dir,
        )
        if clone.returncode:
            raise RuntimeError(f"checkout clone failed: {clone.stderr[-2000:]}")
        # Several benchmark repositories contain legal paths longer than the
        # legacy Windows 260-character limit. Scope the opt-in to this checkout.
        longpaths = self._run(["git", "config", "core.longpaths", "true"], cwd=repository)
        if longpaths.returncode:
            raise RuntimeError(f"could not enable long paths: {longpaths.stderr[-2000:]}")
        fetch = self._run(["git", "fetch", "--depth", "1", "origin", revision], cwd=repository)
        if fetch.returncode:
            raise RuntimeError(f"checkout fetch failed: {fetch.stderr[-2000:]}")
        checkout = self._run(["git", "checkout", "--detach", revision], cwd=repository)
        if checkout.returncode:
            raise RuntimeError(f"checkout failed: {checkout.stderr[-2000:]}")
        self._install_test_artifacts(repository, env.workspace_dir, contract.get("test_materialization") or ())
        head = self._run(["git", "rev-parse", "HEAD"], cwd=repository)
        resolved_head = head.stdout.strip()
        write_json(
            marker_path,
            {
                "requested_revision": revision,
                "resolved_revision": resolved_head,
                "repository": "_external/repository",
                "test_materialization_complete": True,
            },
        )
        return {"repository": str(repository), "revision": resolved_head, "reused": False}

    @staticmethod
    def _install_test_artifacts(repository: Path, workspace: Path, items) -> None:
        """Install typed reproducer artifacts; source snapshots are not patches."""
        repository, workspace = repository.resolve(), workspace.resolve()
        for item in items:
            kind = str(item.get("kind", "test_patch"))
            target = None
            if item.get("workspace_path"):
                target = (repository / str(item["workspace_path"])).resolve()
                target.relative_to(repository)
            if kind == "checkout_test_file":
                if target is None or not target.is_file():
                    raise RuntimeError("Required checkout test file does not exist")
                continue
            public_path = (workspace / str(item["public_path"])).resolve()
            public_path.relative_to(workspace)
            # Standalone artifacts stay in the executor-visible packet workspace.
            # The invocation below points at them directly; copying them into the
            # repository would create a second, divergent source of truth.
            if kind in {"test_cases", "test_runner"}:
                if not public_path.is_file():
                    raise RuntimeError(f"Required standalone test artifact is missing: {public_path}")
                continue
            if kind in {"checkout_test_snapshot", "added_test_file"}:
                if target is None:
                    raise RuntimeError("Test snapshot requires workspace_path")
                if kind == "added_test_file" and target.exists():
                    raise RuntimeError("Added test would overwrite an existing checkout file")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(public_path, target)
                continue
            if kind != "test_patch":
                raise RuntimeError(f"Unsupported test materialization kind: {kind}")
            check = ExternalCodeDebuggingAdapter._run(["git", "apply", "--check", str(public_path)], cwd=repository)
            if check.returncode:
                raise RuntimeError(f"test patch check failed: {check.stderr[-2000:]}")
            applied = ExternalCodeDebuggingAdapter._run(["git", "apply", str(public_path)], cwd=repository)
            if applied.returncode:
                raise RuntimeError(f"test patch apply failed: {applied.stderr[-2000:]}")

    def _quixbugs_invocation(
        self,
        env: EnvironmentHandle,
        python: str,
    ) -> tuple[list[str], Path, dict[str, str]]:
        contract = self._contract(env)
        provenance_path = env.workspace_dir / "fixtures/upstream_provenance.json"
        if not provenance_path.is_file():
            raise RuntimeError("QuixBugs contract is missing upstream provenance")
        provenance = read_json(provenance_path)
        source_path = str((provenance.get("source_paths") or {}).get("buggy") or "")
        if not source_path:
            raise RuntimeError("QuixBugs provenance has no buggy source path")
        source = (self._repository(env) / source_path).resolve()
        source.relative_to(self._repository(env).resolve())
        runner_item = next(
            (item for item in contract.get("test_materialization") or () if item.get("kind") == "test_runner"),
            None,
        )
        case_item = next(
            (item for item in contract.get("test_materialization") or () if item.get("kind") == "test_cases"),
            None,
        )
        if not runner_item or not case_item:
            raise RuntimeError("Standalone runner contract requires test_runner and test_cases")
        runner = (env.workspace_dir / str(runner_item["public_path"])).resolve()
        cases = (env.workspace_dir / str(case_item["public_path"])).resolve()
        runner.relative_to(env.workspace_dir.resolve())
        cases.relative_to(env.workspace_dir.resolve())
        function_name = source.stem
        return (
            [python, str(runner), "--module", str(source), "--function", function_name, "--cases", str(cases)],
            self._repository(env),
            {
                str(key): str(value)
                for key, value in dict(self._runtime_defaults.get("environment") or {}).items()
            },
        )

    def _swebench_invocation(
        self,
        env: EnvironmentHandle,
        python: str,
        arguments: Mapping[str, Any],
    ) -> tuple[list[str], Path, dict[str, str]]:
        command = self._runtime_defaults.get("test_command")
        if isinstance(command, str):
            command = command.strip().split()
        if not command:
            raise RuntimeError(
                "SWE-bench requires an evaluator-provided official harness command; "
                "local pytest inference is intentionally disabled"
            )
        normalized = [python if str(item) in {"python", "python3"} else str(item) for item in command]
        repository = self._repository(env)
        runtime_environment = {
            str(key): str(value)
            for key, value in dict(self._runtime_defaults.get("environment") or {}).items()
        }
        existing = runtime_environment.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        pythonpath = str(repository) if not existing else str(repository) + os.pathsep + existing
        runtime_environment["PYTHONPATH"] = pythonpath
        return normalized, repository, runtime_environment

    def _test_invocation(
        self,
        env: EnvironmentHandle,
        arguments: Mapping[str, Any],
    ) -> tuple[list[str], Path, dict[str, str]]:
        contract = self._contract(env)
        python = str(
            self._runtime_defaults.get("python_executable")
            or os.environ.get(self.python_environment_variable)
            or sys.executable
        )
        interface = str(contract.get("test_interface") or "")
        if interface == "standalone_jsonl_runner":
            return self._quixbugs_invocation(env, python)
        if interface == "official_swebench_harness":
            return self._swebench_invocation(env, python, arguments)
        test_ids = [str(item) for item in contract.get("designated_test_ids") or ()]
        if not test_ids:
            raise RuntimeError("The public execution contract has no designated_test_ids")
        if contract.get("test_framework") == "unittest":
            command = [python, "-m", "unittest", "-q", *test_ids]
        else:
            command = [python, "-m", "pytest", "-q", *test_ids]
        repository = self._repository(env)
        runtime_environment = {
            str(key): str(value)
            for key, value in dict(self._runtime_defaults.get("environment") or {}).items()
        }
        existing = runtime_environment.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        pythonpath = str(repository) if not existing else str(repository) + os.pathsep + existing
        runtime_environment["PYTHONPATH"] = pythonpath
        return command, repository, runtime_environment

    def _test_command(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> list[str]:
        command, _, _ = self._test_invocation(env, arguments)
        return command

    def _run_tests(
        self,
        env: EnvironmentHandle,
        arguments: Mapping[str, Any],
        *,
        scope: str,
    ) -> ToolResult:
        state = self._state(env)
        command, cwd, environment = self._test_invocation(env, arguments)
        try:
            completed = self._run(command, cwd=cwd, timeout=600, environment=environment)
        except subprocess.TimeoutExpired as error:
            outcome_key = "baseline_test_outcome" if scope == "baseline" else "candidate_test_outcome"
            state[outcome_key] = "error"
            filename = "baseline_test_result.json" if scope == "baseline" else "candidate_test_result.json"
            artifact_key = "baseline_test_result" if scope == "baseline" else "candidate_test_result"
            write_json(
                env.workspace_dir / "_runtime" / filename,
                {
                    "command": command,
                    "cwd": str(cwd),
                    "environment_keys": sorted(environment),
                    "return_code": None,
                    "raw_output": f"{type(error).__name__}: {error}",
                    "outcome": "error",
                    "scope": scope,
                    "fresh": True,
                    "failure_kind": "environment",
                },
            )
            state.setdefault("artifacts", {})[artifact_key] = {
                "artifact_type": "verification_record",
                "path": f"_runtime/{filename}",
                "status": "error",
                "summary": f"Fresh {scope} regression execution timed out.",
            }
            self._save_state(env, state)
            return ToolResult(
                "produce_candidate" if scope == "baseline" else "validate_candidate",
                "timeout",
                {"error": f"{type(error).__name__}: {error}", "scope": scope},
                "The concrete regression command timed out.",
            )
        raw_output = (completed.stdout + completed.stderr)[-self.max_output_chars :]
        passed = completed.returncode == 0
        summary = {
            "outcome": "pass" if passed else "fail",
            "scope": scope,
            "fresh": True,
            "failure_kind": None if passed else "behavior",
        }
        outcome_key = "baseline_test_outcome" if scope == "baseline" else "candidate_test_outcome"
        state[outcome_key] = summary["outcome"]
        if scope == "candidate":
            state['validated_version']=self.evidence_version(env) if passed else None
            state["stage"] = "verified" if passed else "candidate_invalid"
            state["observable_issues"] = [] if passed else ["The executor-visible regression contract failed."]
        elif not passed:
            state["stage"] = "baseline_reproduced"
            state["observable_issues"] = ["The supplied baseline fails the regression contract."]
        else:
            state["stage"] = "initial"
            state["observable_issues"] = ["The supplied regression did not reproduce the reported defect."]
        filename = "baseline_test_result.json" if scope == "baseline" else "candidate_test_result.json"
        artifact_key = "baseline_test_result" if scope == "baseline" else "candidate_test_result"
        write_json(
            env.workspace_dir / "_runtime" / filename,
            {
                "command": command,
                "cwd": str(cwd),
                "environment_keys": sorted(environment),
                "return_code": completed.returncode,
                "raw_output": raw_output,
                **summary,
            },
        )
        state.setdefault("artifacts", {})[artifact_key] = {
            "artifact_type": "verification_record",
            "path": f"_runtime/{filename}",
            "status": "available",
            "passed": passed,
            "summary": f"Fresh {scope} regression result: {summary['outcome']}.",
            "evidence_refs": [],
        }
        self._save_state(env, state)
        return ToolResult(
            "produce_candidate" if scope == "baseline" else "validate_candidate",
            "success" if passed else "invalid_output",
            {"test_state": summary, "return_code": completed.returncode, "raw_output": raw_output},
            "The regression contract passed." if passed else "The regression contract reproduced a failure.",
        )

    def _produce_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        if self._state(env).get('artifacts',{}).get('candidate',{}).get('status')=='applied':
            return ToolResult('produce_candidate','rejected',{},'Baseline reproduction cannot discard or relabel an applied candidate.')
        try:
            checkout = self._materialize_checkout(env)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            state = self._state(env)
            state["checkout_ready"] = False
            state["baseline_test_outcome"] = "error"
            failure = {
                "outcome": "error",
                "scope": "baseline",
                "fresh": True,
                "failure_kind": "environment",
                "error": f"{type(error).__name__}: {error}",
            }
            write_json(env.workspace_dir / "_runtime/baseline_test_result.json", failure)
            state.setdefault("artifacts", {})["baseline_test_result"] = {
                "artifact_type": "verification_record",
                "path": "_runtime/baseline_test_result.json",
                "status": "error",
                "summary": "Pinned checkout materialization failed.",
                "evidence_refs": [],
            }
            self._save_state(env, state)
            return ToolResult(
                "produce_candidate",
                "tool_failure",
                {"error": failure["error"], "test_state": failure},
                "The pinned repository checkout could not be prepared.",
            )
        state = self._state(env)
        state["checkout_ready"] = True
        state.setdefault("artifacts", {})["repository_checkout"] = {
            "artifact_type": "repository_checkout",
            "path": "_external/repository",
            "status": "available",
            "revision": checkout["revision"],
        }
        self._save_state(env, state)
        result = self._run_tests(env, arguments, scope="baseline")
        outcome = str((result.output.get("test_state") or {}).get("outcome", ""))
        if outcome == "fail":
            # A failing baseline is the successful product of the reproducer;
            # invalid_output is reserved for a candidate patch that fails.
            status = "success"
        elif outcome == "pass":
            state = self._state(env)
            status = "no_progress"
        else:
            status = result.status
        return ToolResult(
            "produce_candidate",
            status,
            {"checkout": checkout, **dict(result.output)},
            result.observable_message,
            result.cost,
        )

    def _diagnose_failure(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        result_path = env.workspace_dir / "_runtime/candidate_test_result.json"
        evidence_ref = "candidate-test-result"
        if not result_path.exists():
            result_path = env.workspace_dir / "_runtime/baseline_test_result.json"
            evidence_ref = "baseline-test-result"
        if not result_path.exists():
            return ToolResult("diagnose_failure", "no_progress", {}, "No executed reproducer is available.", 0.0)
        result = read_json(result_path)
        analysis = str(arguments.get("analysis") or "").strip()
        if not analysis:
            return ToolResult(
                "diagnose_failure",
                "invalid_output",
                {"evidence_ref": evidence_ref},
                "No executor-authored diagnosis was supplied.",
                0.0,
            )
        diagnosis = {
            "analysis": analysis,
            "evidence_refs": [evidence_ref],
            "source_outcome": result.get("outcome"),
        }
        write_json(env.workspace_dir / "_runtime/diagnosis.json", diagnosis)
        state.setdefault("artifacts", {})["diagnosis"] = {
            "artifact_type": "diagnosis",
            "path": "_runtime/diagnosis.json",
            "status": "candidate",
            "summary": "Executor-authored diagnosis grounded in concrete test output.",
            "evidence_refs": [evidence_ref],
        }
        state["stage"] = "diagnosed"
        self._save_state(env, state)
        return ToolResult(
            "diagnose_failure",
            "success",
            diagnosis,
            "The executor-authored diagnosis was recorded.",
        )

    def _repair_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        if not self._repository(env).is_dir():
            return ToolResult('repair_candidate','rejected',{},'No pinned repository checkout is available.')
        patch = str(arguments.get("patch") or "")
        if not patch.strip():
            return ToolResult(
                "repair_candidate",
                "no_progress",
                {},
                "No executor-authored patch was supplied.",
                0.0,
            )
        prior=self._state(env)
        had_applied=prior.get('artifacts',{}).get('candidate',{}).get('status')=='applied'
        canonical=env.workspace_dir/'_runtime/candidate.patch'
        patch_path=(env.workspace_dir/f'_runtime/attempts/{len(prior.get("tool_events",[])):04}_candidate.patch') if had_applied else canonical
        attempt_path=patch_path
        patch_path.parent.mkdir(parents=True,exist_ok=True)
        # Unified diffs are byte-sensitive. Force LF even on Windows; text-mode
        # newline translation can make otherwise valid multi-file patches look
        # corrupt to git apply.
        with patch_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(patch if patch.endswith("\n") else patch + "\n")
        repository = self._repository(env)
        strategies: list[tuple[Path, list[str], str]] = [(patch_path, [], "repository_relative")]
        contract = self._contract(env)
        if str(contract.get("test_interface") or "") == "standalone_jsonl_runner":
            provenance = read_json(env.workspace_dir / "fixtures/upstream_provenance.json")
            source_path = Path(str((provenance.get("source_paths") or {}).get("buggy") or ""))
            # QuixBugs publishes the minimal diff against a single-file executor
            # view (e.g. sieve.py), while the pinned repository stores that file
            # under python_programs/. Apply the same diff at that explicit root.
            apply_prefix = [f"--directory={source_path.parent.as_posix()}"]
            public_name = f"public/repo/{source_path.name}"
            normalized_text = patch.replace(f"a/{public_name}", f"a/{source_path.name}").replace(
                f"b/{public_name}", f"b/{source_path.name}"
            )
            normalized_path = patch_path.with_name(patch_path.stem+'_repository.patch') if had_applied else env.workspace_dir / "_runtime/candidate_repository.patch"
            with normalized_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(normalized_text if normalized_text.endswith("\n") else normalized_text + "\n")
            strategies = [
                (normalized_path, apply_prefix, "public_view_to_pinned_source"),
                (patch_path, apply_prefix, "single_file_to_pinned_source"),
                (patch_path, [], "repository_relative"),
            ]
        selected = None
        errors = []
        for candidate_path, apply_prefix, normalization in strategies:
            check = self._run(
                ["git", "apply", *apply_prefix, "--check", str(candidate_path)],
                cwd=repository,
            )
            if check.returncode == 0:
                selected = (candidate_path, apply_prefix, normalization)
                break
            errors.append(check.stderr[-2000:])
        if selected is None:
            if had_applied:
                self._preserve_rejected_attempt(env,patch_path,'New patch rejected by git apply --check; current applied candidate is unchanged.')
                return ToolResult('repair_candidate','invalid_output',{'error':'\n'.join(errors),'attempt_path':patch_path.relative_to(env.workspace_dir).as_posix()},'New patch rejected; current applied candidate is unchanged.')
            state = self._state(env)
            state["candidate_patch_outcome"] = "rejected"
            state["observable_issues"] = [
                "The executor-authored patch was rejected by git apply --check."
            ]
            state.setdefault("artifacts", {})["candidate"] = {
                "artifact_type": "candidate_patch",
                "path": "_runtime/candidate.patch",
                "sha256": sha256_file(patch_path),
                "status": "invalid",
                "summary": "Executor-authored patch rejected by git apply --check.",
            }
            self._save_state(env, state)
            error_detail = "\n---\n".join(errors)[-6000:]
            compact_error = " ".join(error_detail.split())[-1200:]
            return ToolResult(
                "repair_candidate",
                "invalid_output",
                {
                    "error": error_detail,
                    "failure_type": "patch_apply_rejected",
                },
                (
                    "The proposed patch was rejected by git apply --check. "
                    f"git reported: {compact_error}"
                    if compact_error
                    else "The proposed patch was rejected by git apply --check."
                ),
            )
        selected_path, apply_prefix, normalization = selected
        paths=self._run(['git','apply',*apply_prefix,'--numstat','-z',str(selected_path)],cwd=repository)
        names=[record.split('\t',2)[-1] for record in paths.stdout.split('\0') if record]
        if paths.returncode or not names or any(not self.patch_path_allowed(name) for name in names):
            if had_applied:
                self._preserve_rejected_attempt(env,patch_path,'New patch touches protected paths; current applied candidate is unchanged.')
            return ToolResult('repair_candidate','invalid_output',{'rejected_paths':names},'Patch may only modify implementation Python files; tests, harness/config and reference files are protected.')
        applied = self._run(["git", "apply", *apply_prefix, str(selected_path)], cwd=repository)
        if applied.returncode:
            if had_applied:
                self._preserve_rejected_attempt(env,patch_path,'New patch could not be applied; current applied candidate is unchanged.')
                return ToolResult('repair_candidate','tool_failure',{'error':applied.stderr},'New patch could not be applied; current applied candidate is unchanged.')
            state = self._state(env)
            state["candidate_patch_outcome"] = "rejected"
            state["observable_issues"] = [
                "The checked patch could not be applied to the pinned checkout."
            ]
            state.setdefault("artifacts", {})["candidate"] = {
                "artifact_type": "candidate_patch",
                "path": "_runtime/candidate.patch",
                "sha256": sha256_file(patch_path),
                "status": "invalid",
                "summary": "Patch passed the check but failed during application.",
            }
            self._save_state(env, state)
            return ToolResult(
                "repair_candidate",
                "tool_failure",
                {
                    "error": applied.stderr[-2000:],
                    "failure_type": "patch_application_failed",
                },
                "The checked patch could not be applied.",
            )
        if had_applied:
            history=env.workspace_dir/f'_runtime/applied_history/{prior["artifacts"]["candidate"]["sha256"]}.patch'
            history.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(canonical,history)
            canonical.write_text(self._cumulative_patch(env),encoding='utf-8',newline='\n')
            prior.setdefault('candidate_history',[]).append({**prior['artifacts']['candidate'],'path':history.relative_to(env.workspace_dir).as_posix()})
            self._save_state(env,prior)
        patch_path=canonical
        state = self._state(env)
        state["stage"] = "candidate"
        state["observable_issues"] = []
        state["candidate_patch_outcome"] = "applied"
        state['validated_version']=None
        state['terminal']=False
        state["candidate_test_outcome"] = "not_run"
        state.pop("final_artifact_ready", None)
        state["final_artifact_ready"] = False
        state.setdefault("artifacts", {}).pop("candidate_test_result", None)
        state.setdefault("artifacts", {}).pop("downstream", None)
        state.setdefault("artifacts", {}).pop("final", None)
        state.setdefault("artifacts", {})["candidate"] = {
            "artifact_type": "candidate_patch",
            "path": "_runtime/candidate.patch",
            "sha256": sha256_file(patch_path),
            "status": "applied",
            "summary": "Executor-authored patch applied successfully; validation is pending.",
        }
        self._save_state(env, state)
        return ToolResult(
            "repair_candidate",
            "success",
            {"patch_sha256": sha256_file(patch_path), "path_normalization": normalization,
             **({'patch_composition':'cumulative_repository_diff','attempt_path':attempt_path.relative_to(env.workspace_dir).as_posix(),
                 'previous_applied_sha256':prior['artifacts']['candidate']['sha256']} if had_applied else {})},
            "The executor-authored patch was applied to the pinned checkout.",
        )

    def _validate_candidate(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        state = self._state(env)
        if str(state.get("candidate_patch_outcome", "absent")) != "applied":
            return ToolResult(
                "validate_candidate",
                "rejected",
                {},
                "No applied candidate patch is available for validation.",
                0.0,
            )
        return self._run_tests(env, arguments, scope="candidate")

    def _execute_downstream(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        state = self._state(env)
        if str(state.get("candidate_test_outcome")) != "pass":
            return ToolResult(
                "execute_downstream",
                "rejected",
                {},
                "A fresh passing regression result is required.",
                0.0,
            )
        downstream = {
            "verified_patch": "_runtime/candidate.patch",
            "test_result": "_runtime/candidate_test_result.json",
            "postcondition_observed": True,
        }
        write_json(env.workspace_dir / "_runtime/downstream.json", downstream)
        state["stage"] = "downstream"
        state["final_artifact_ready"] = True
        state.setdefault("artifacts", {})["downstream"] = {
            "artifact_type": "verified_patch",
            "path": "_runtime/downstream.json",
            "status": "available",
            "summary": "Candidate patch promoted after a fresh passing regression result.",
            "evidence_refs": ["candidate-patch", "candidate-test-result"],
        }
        self._save_state(env, state)
        return ToolResult(
            "execute_downstream",
            "success",
            {"downstream": downstream},
            "The passing patch was promoted to a verified downstream artifact.",
        )

    def _finalize(self, env: EnvironmentHandle, arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        state = self._state(env)
        downstream_path = env.workspace_dir / "_runtime/downstream.json"
        if not downstream_path.exists() or not bool(state.get("final_artifact_ready")):
            return ToolResult(
                "finalize",
                "rejected",
                {},
                "A verified final-ready artifact is required before commitment.",
                0.0,
            )
        final = {
            "committed": True,
            "result": read_json(downstream_path),
            "base_workflow_id": env.manifest.base_workflow_id,
        }
        path = env.workspace_dir / "_runtime/final.json"
        write_json(path, final)
        state.setdefault("artifacts", {})["final"] = {
            "artifact_type": "committed_final_artifact",
            "path": "_runtime/final.json",
            "sha256": sha256_file(path),
            "status": "committed",
            "summary": "Final artifact committed from the verified patch.",
            "evidence_refs": ["verified-patch", "candidate-test-result"],
        }
        state["stage"] = "complete"
        state["terminal"] = True
        state["final_artifact_ready"] = True
        self._save_state(env, state)
        return ToolResult("finalize", "success", final, "The final artifact was committed.", 0.0)
