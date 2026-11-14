from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DOMAIN_LABELS = {
    "database_migration_integrity": "database migration",
    "code_debugging": "code debugging",
    "formal_proof": "formal proof",
    "structured_data_processing": "structured data",
}


def load_observation(state: dict[str, Any]) -> dict[str, Any]:
    observation = json.loads(state["router_input_json"])
    if not isinstance(observation, dict):
        raise ValueError("router_input_json must decode to an object")
    return observation


def decision_step(decision_id: str) -> str:
    return decision_id.rsplit(":", 1)[-1]


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " "))


def candidate_actions(state: dict[str, Any], observation: dict[str, Any]) -> list[str]:
    state_actions = json.loads(state["candidate_actions_json"])
    visible_actions = (observation.get("constraints") or {}).get("allowed_actions")
    if visible_actions is not None and state_actions != visible_actions:
        raise ValueError("candidate_actions_json does not match router-visible allowed_actions")
    return state_actions


def role_cards(observation: dict[str, Any]) -> list[dict[str, Any]]:
    cards = observation.get("agent_profiles") or []
    if not isinstance(cards, list):
        raise ValueError("agent_profiles must be a list")
    return cards


def role_pool_is_complete(state: dict[str, Any], observation: dict[str, Any]) -> bool:
    actions = candidate_actions(state, observation)
    role_ids = [card.get("agent_id") for card in role_cards(observation)]
    return actions == [*role_ids, "STOP"]


def contract_badges(domain: str, observation: dict[str, Any]) -> list[str]:
    current = observation.get("current_state") or {}
    if domain == "database_migration_integrity":
        contract = current.get("target_contract") or {}
        return [
            f"{len(contract.get('checks') or [])} target checks",
            f"{len(contract.get('behavioral_checks') or [])} behavioral checks",
            f"{len(contract.get('required_integrity_checks') or [])} integrity checks",
            "reference migration visible" if contract.get("reference_migration_visible") else "reference migration hidden",
        ]
    if domain == "code_debugging":
        contract = current.get("execution_contract") or {}
        tests = contract.get("designated_test_ids") or []
        return [
            f"{len(tests)} designated test{'s' if len(tests) != 1 else ''}",
            f"framework: {contract.get('test_framework', 'unknown')}",
            "network enabled" if contract.get("network_during_test_execution") else "network disabled",
            "oracle hidden" if contract.get("evaluator_oracle_visibility") == "private" else "oracle visibility declared",
        ]
    if domain == "formal_proof":
        contract = current.get("public_execution_contract") or {}
        edit_contract = current.get("edit_contract") or {}
        return [
            f"engine: {contract.get('proof_engine') or contract.get('language') or 'Lean'}",
            f"edit scope: {edit_contract.get('scope') or edit_contract.get('allowed') or 'declared'}",
            "network enabled" if contract.get("network_during_execution") else "network disabled",
            "oracle hidden" if contract.get("evaluator_oracle_visibility") == "private" else "public verification contract",
        ]
    if domain == "structured_data_processing":
        contract = current.get("candidate_contract") or {}
        public_task = current.get("public_task") or {}
        return [
            f"candidate fields: {len(contract)}",
            f"task fields: {len(public_task)}",
            "reference hidden",
            "public validation required",
        ]
    return [f"benchmark: {observation.get('benchmark_version', 'unknown')}"]


def terminal_summary(domain: str, observation: dict[str, Any]) -> tuple[bool, str]:
    current = observation.get("current_state") or {}
    if "terminal" in current:
        terminal = bool(current.get("terminal"))
        if terminal:
            return True, "Router-visible terminal flag is true"
        if domain == "database_migration_integrity":
            validation = _artifact(current, "validation")
            return False, "Validation chưa thành công" if not validation else "Terminal flag vẫn false"
        if domain == "formal_proof":
            return False, "Proof chưa compile và verify đầy đủ"
        if domain == "structured_data_processing":
            return False, "Public execution/validation chưa hoàn tất"
        return False, "Terminal flag is false"
    if domain == "code_debugging":
        complete = current.get("task_status") in {"complete", "completed", "terminal"}
        return complete, str(current.get("terminal_contract") or current.get("task_status") or "Not terminal")
    return False, "No terminal signal is visible"


def current_facts(domain: str, observation: dict[str, Any]) -> list[dict[str, str]]:
    current = observation.get("current_state") or {}
    if domain == "database_migration_integrity":
        execution = current.get("latest_execution") or {}
        actor = current.get("latest_actor_result") or {}
        stdout = ((execution.get("result") or {}).get("stdout"))
        database_changed = execution.get("database_changed")
        return [
            _artifact_fact(current, "inspection", "Schema inspection"),
            _artifact_fact(current, "profile", "Data profile"),
            _artifact_fact(current, "candidate", "SQL candidate"),
            {
                "label": "Latest execution",
                "value": _display(actor.get("status") or _execution_status(execution)),
                "path": "current_state.latest_actor_result.status / current_state.latest_execution.result.returncode",
            },
            {
                "label": "Observed failure",
                "value": _display(stdout or "missing"),
                "path": "current_state.latest_execution.result.stdout",
            },
            {
                "label": "Database changed",
                "value": _display(database_changed if database_changed is not None else "unknown"),
                "path": "current_state.latest_execution.database_changed",
            },
            _artifact_fact(current, "validation", "Validation artifact"),
            {
                "label": "Terminal",
                "value": _display(current.get("terminal")),
                "path": "current_state.terminal",
            },
        ]
    if domain == "code_debugging":
        facts = []
        for index, fact in enumerate(current.get("facts") or []):
            facts.append(
                {
                    "label": str(fact.get("key", f"Fact {index + 1}")).replace("_", " ").title(),
                    "value": _display(fact.get("value")),
                    "path": f"current_state.facts[{index}].value",
                }
            )
        facts.extend(
            [
                {
                    "label": "Open requirements",
                    "value": str(len(current.get("open_requirements") or [])),
                    "path": "current_state.open_requirements",
                },
                {
                    "label": "Issues",
                    "value": str(len(current.get("issues") or [])),
                    "path": "current_state.issues",
                },
                {
                    "label": "Task status",
                    "value": _display(current.get("task_status")),
                    "path": "current_state.task_status",
                },
            ]
        )
        return facts
    if domain == "formal_proof":
        execution = current.get("latest_execution") or {}
        feedback = current.get("latest_tool_feedback") or {}
        return [
            _artifact_fact(current, "candidate", "Proof candidate"),
            _artifact_fact(current, "compilation", "Compilation artifact"),
            _artifact_fact(current, "diagnosis", "Diagnosis artifact"),
            _artifact_fact(current, "verification", "Verification artifact"),
            {
                "label": "Latest execution",
                "value": _display(_execution_status(execution)),
                "path": "current_state.latest_execution.result.returncode",
            },
            {
                "label": "Tool feedback",
                "value": _display(feedback.get("status") or feedback.get("summary") or ("available" if feedback else "missing")),
                "path": "current_state.latest_tool_feedback",
            },
            {
                "label": "Terminal",
                "value": _display(current.get("terminal")),
                "path": "current_state.terminal",
            },
        ]
    if domain == "structured_data_processing":
        execution = current.get("latest_execution") or {}
        return [
            _artifact_fact(current, "workbook_inventory", "Workbook inventory"),
            _artifact_fact(current, "data_profile", "Data profile"),
            _artifact_fact(current, "candidate_solution", "Candidate solution"),
            _artifact_fact(current, "public_execution", "Public execution"),
            _artifact_fact(current, "public_validation", "Public validation"),
            {
                "label": "Latest execution",
                "value": _display(execution.get("status") or (execution.get("summary") or {}).get("returncode") or "missing"),
                "path": "current_state.latest_execution",
            },
            {
                "label": "Rejected attempts",
                "value": _display(current.get("rejected_attempt_count", 0)),
                "path": "current_state.rejected_attempt_count",
            },
            {
                "label": "Terminal",
                "value": _display(current.get("terminal")),
                "path": "current_state.terminal",
            },
        ]
    return _generic_scalar_facts(current)


def artifact_items(observation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    artifacts = (observation.get("current_state") or {}).get("artifacts") or {}
    if isinstance(artifacts, dict):
        return [(str(name), value if isinstance(value, dict) else {"value": value}) for name, value in artifacts.items()]
    if isinstance(artifacts, list):
        return [
            (str(item.get("artifact_id") or item.get("type") or f"artifact-{index}"), item)
            for index, item in enumerate(artifacts)
            if isinstance(item, dict)
        ]
    return [("artifacts", {"value": artifacts})]


def visible_sections(observation: dict[str, Any]) -> dict[str, Any]:
    """Account for every top-level router-visible field in the review UI."""
    return {
        "decision_brief": {
            "task": observation.get("task"),
            "budget": observation.get("budget"),
            "current_state": observation.get("current_state"),
        },
        "role_cards": observation.get("agent_profiles"),
        "action_review": observation.get("constraints"),
        "evidence": observation.get("recent_events"),
        "audit_json": observation,
    }


def discover_labeling_dirs(root: Path, labeling_dir: str) -> list[Path]:
    if labeling_dir != "all":
        path = root / labeling_dir
        return [path] if path.is_dir() else []
    return sorted(
        path
        for path in root.glob("labeling*")
        if path.is_dir() and (path / "sample_manifest.json").exists()
    )


def _artifact(current: dict[str, Any], name: str) -> Any:
    artifacts = current.get("artifacts") or {}
    if isinstance(artifacts, dict):
        return artifacts.get(name)
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if name in {artifact.get("artifact_id"), artifact.get("type")}:
                return artifact
    return None


def _artifact_fact(current: dict[str, Any], name: str, label: str) -> dict[str, str]:
    artifact = _artifact(current, name)
    if artifact is None:
        value = "missing"
        path = f"current_state.artifacts.{name} (absent)"
    elif isinstance(artifact, dict):
        freshness = artifact.get("fresh")
        value = "fresh" if freshness is True else str(artifact.get("status") or "available")
        path = f"current_state.artifacts.{name}"
    else:
        value = "available"
        path = f"current_state.artifacts.{name}"
    return {"label": label, "value": value, "path": path}


def _execution_status(execution: dict[str, Any]) -> str:
    if not execution:
        return "missing"
    if execution.get("status") is not None:
        return str(execution["status"])
    result = execution.get("result") or {}
    returncode = result.get("returncode")
    if returncode is None:
        return "available"
    return "success" if returncode == 0 else "failed"


def _display(value: Any, limit: int = 420) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "missing"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _generic_scalar_facts(current: dict[str, Any]) -> list[dict[str, str]]:
    facts = []
    for key, value in current.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            facts.append(
                {
                    "label": key.replace("_", " ").title(),
                    "value": _display(value),
                    "path": f"current_state.{key}",
                }
            )
    return facts
